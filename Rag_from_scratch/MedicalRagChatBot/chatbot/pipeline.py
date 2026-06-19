from pathlib import Path
import json
import logging
import logging.handlers
import sys
import os

from langchain_core.documents         import Document
from langchain_openai                 import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers    import StrOutputParser
from langchain_core.runnables         import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from chatbot.ingestion.parser         import PDFParser
from chatbot.ingestion.builder        import DocumentBuilder
from chatbot.retrieval.vector_store   import VectorStoreManager
from chatbot.retrieval.retriever      import SmartRetriever
from chatbot.session.session          import SessionManager
from chatbot.chain.prompt             import MEDICAL_PROMPT

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config


LOG_DIR = Path("./logs")


# =============================================================================
# Logging
# =============================================================================

def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_h = logging.handlers.RotatingFileHandler(
        LOG_DIR / "pipeline.log",
        maxBytes    = 10 * 1024 * 1024,
        backupCount = 5,
        encoding    = "utf-8",
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_h)

    for lib in ["urllib3", "httpx", "openai", "chromadb", "docling"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


log = logging.getLogger("pipeline")


# =============================================================================
# Helpers
# =============================================================================

def _get_pdf_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        if source_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {source_path}")
        return [source_path]
    if source_path.is_dir():
        files = sorted(source_path.rglob("*.pdf"))
        if not files:
            raise ValueError(f"No PDFs in: {source_path}")
        return files
    raise ValueError(f"Path not found: {source_path}")


def _header(title: str):
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# =============================================================================
# Stage 1 — Parse: PDF -> JSON
# =============================================================================

def flow_parse(
    source_path: Path,
    cache_dir:   Path = Path("./json_cache"),
) -> dict:
    """
    Stage 1: Parse PDFs to JSON.
    Zero LLM. Zero embeddings.
    Safe to re-run — uses cache.

    Returns:
        {
            "parsed":  int,
            "skipped": int,
            "reports": {filename: report_dict}
        }
    """
    _header("STAGE 1: Parse PDFs -> JSON")
    log.info(f"Source    : {source_path}")
    log.info(f"Cache dir : {cache_dir}")

    try:
        pdf_files = _get_pdf_files(Path(source_path))
    except ValueError as e:
        log.error(str(e))
        return {"parsed": 0, "skipped": 0, "reports": {}}

    log.info(f"Found {len(pdf_files)} PDF(s)")

    parser  = PDFParser(cache_dir=Path(cache_dir))
    reports = {}
    skipped = []

    for i, pdf_file in enumerate(pdf_files, 1):
        log.info(f"  [{i}/{len(pdf_files)}] {pdf_file.name}")
        try:
            report = parser.parse(pdf_file)

            total    = sum(len(t) for t in report.get("sections", {}).values())
            sections = list(report.get("sections", {}).keys())

            log.info(f"    date     : {report.get('report_date', '?')}")
            log.info(f"    sections : {sections}")
            log.info(f"    tests    : {total}")

            if total == 0:
                log.warning(f"    No tests found — skipping")
                skipped.append(pdf_file.name)
            else:
                reports[pdf_file.name] = report

        except Exception as e:
            log.error(f"    Failed: {e}")
            skipped.append(pdf_file.name)

    log.info("-" * 60)
    log.info(f"Parsed  : {len(reports)}")
    log.info(f"Skipped : {len(skipped)}")
    if skipped:
        log.warning(f"Skipped files: {skipped}")

    return {
        "parsed":  len(reports),
        "skipped": len(skipped),
        "reports": reports,
    }


# =============================================================================
# Stage 2 — Embed: JSON -> ChromaDB
# =============================================================================

def flow_embed(
    cache_dir:   Path = Path("./json_cache"),
    persist_dir: str  = "./chroma_medical_db",
    force:       bool = False,
) -> dict:
    """
    Stage 2: Embed cached JSON into ChromaDB.
    Reads from cache. No LLM.
    Pass force=True to re-embed even if vectors exist.

    Returns:
        {
            "total":   int,
            "skipped": bool
        }
    """
    _header("STAGE 2: Embed JSON -> ChromaDB")
    log.info(f"Cache dir   : {cache_dir}")
    log.info(f"Persist dir : {persist_dir}")
    log.info(f"Force       : {force}")

    config = load_config()

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint   = config["AZURE_OPENAI_ENDPOINT"],
        api_key          = config["AZURE_OPENAI_API_KEY"],
        azure_deployment = config["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
        api_version      = config["AZURE_OPENAI_EMBEDDINGS_API_VERSION"],
    )

    vs = VectorStoreManager(embeddings=embeddings, persist_dir=persist_dir)

    if not vs.is_empty() and not force:
        count = vs.count()
        log.info(f"Chroma already has {count} docs — skipping")
        log.info("Use force=True to re-embed")
        return {"total": count, "skipped": True}

    # Load cached reports
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        log.error(f"Cache dir not found: {cache_dir}")
        log.error("Run flow_parse() first")
        return {"total": 0, "skipped": False}

    builder   = DocumentBuilder()
    all_docs  = []
    loaded    = 0

    for json_file in sorted(cache_path.glob("*.json")):
        try:
            report = json.loads(json_file.read_text())
            docs   = builder.build(report)
            all_docs.extend(docs)
            loaded += 1
            log.info(f"  {json_file.name} -> {len(docs)} docs")
        except Exception as e:
            log.error(f"  Failed {json_file.name}: {e}")

    if not all_docs:
        log.error("No documents to embed")
        return {"total": 0, "skipped": False}

    log.info(f"Embedding {len(all_docs)} documents...")
    vs.add(all_docs)

    log.info("-" * 60)
    log.info(f"Loaded    : {loaded} reports")
    log.info(f"Indexed   : {len(all_docs)} documents")

    return {"total": len(all_docs), "skipped": False}


# =============================================================================
# Stage 3 — Chat: ChromaDB -> LLM -> answers
# =============================================================================

def flow_chat(
    persist_dir: str = "./chroma_medical_db",
    session_id:  str = "session_1",
):
    """
    Stage 3: Start chat interface.
    Requires flow_parse + flow_embed to have been run first.
    """
    _header("STAGE 3: Chat")
    log.info(f"Persist dir : {persist_dir}")
    log.info(f"Session     : {session_id}")

    config = load_config()

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint   = config["AZURE_OPENAI_ENDPOINT"],
        api_key          = config["AZURE_OPENAI_API_KEY"],
        azure_deployment = config["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
        api_version      = config["AZURE_OPENAI_EMBEDDINGS_API_VERSION"],
    )

    llm = AzureChatOpenAI(
        azure_endpoint   = config["AZURE_OPENAI_ENDPOINT"],
        api_key          = config["AZURE_OPENAI_API_KEY"],
        azure_deployment = config["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version      = config["AZURE_OPENAI_CHAT_API_VERSION"],
        temperature      = 0,
    )

    vs_manager = VectorStoreManager(embeddings=embeddings, persist_dir=persist_dir)

    if vs_manager.is_empty():
        log.error("No documents in ChromaDB")
        log.error("Run flow_parse() then flow_embed() first")
        return

    count = vs_manager.count()
    log.info(f"Vector store: {count} documents ready")

    retriever       = SmartRetriever(vs_manager)
    session_manager = SessionManager(max_messages=4)

    def get_context(question: str) -> str:
        docs = retriever.retrieve(question)
        retriever.debug(question, docs)
        return retriever.format(docs)

    core_chain = (
        {
            "context":      RunnableLambda(lambda x: get_context(x["question"])),
            "question":     RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | MEDICAL_PROMPT
        | llm
        | StrOutputParser()
    )

    chain = RunnableWithMessageHistory(
        core_chain,
        session_manager.get,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    log.info("Chat ready. Type exit or quit to stop.")
    log.info("-" * 60)

    print("\nMedical Records Assistant")
    print("Ask about tests, results, dates, abnormal findings, or improvements.")
    print("Type exit or quit to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        session_manager.trim(session_id)

        answer = chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"\nAnswer:\n{answer}\n")


# =============================================================================
# Combined flows
# =============================================================================

def flow_ingest(
    source_path: Path,
    cache_dir:   Path = Path("./json_cache"),
    persist_dir: str  = "./chroma_medical_db",
    force:       bool = False,
) -> dict:
    """
    Stage 1 + 2: Parse PDFs + Embed into ChromaDB.
    No chat.
    """
    _header("FLOW: Ingest (Parse + Embed)")

    parse_result = flow_parse(
        source_path = Path(source_path),
        cache_dir   = Path(cache_dir),
    )

    if parse_result["parsed"] == 0:
        log.error("No PDFs parsed. Aborting embed.")
        return {"status": "failed", "parsed": 0, "total": 0}

    embed_result = flow_embed(
        cache_dir   = Path(cache_dir),
        persist_dir = persist_dir,
        force       = force,
    )

    return {
        "status":  "ok",
        "parsed":  parse_result["parsed"],
        "skipped": parse_result["skipped"],
        "total":   embed_result["total"],
    }


def flow_run(
    source_path: Path,
    cache_dir:   Path = Path("./json_cache"),
    persist_dir: str  = "./chroma_medical_db",
    session_id:  str  = "session_1",
    force:       bool = False,
):
    """All stages: Parse + Embed + Chat."""
    _header("FLOW: Full Pipeline")

    result = flow_ingest(
        source_path = source_path,
        cache_dir   = cache_dir,
        persist_dir = persist_dir,
        force       = force,
    )

    if result["status"] != "ok":
        log.error("Ingestion failed. Cannot start chat.")
        return

    flow_chat(
        persist_dir = persist_dir,
        session_id  = session_id,
    )