# Activities = individual units of work (equivalent to @task in Prefect)
# Each activity runs in a worker process
# Automatic retry, timeout, heartbeat built in

from pathlib import Path
from temporalio import activity
from langchain_core.documents import Document
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
from config import load_config

from chatbot.ingestion.parser       import PDFParser
from chatbot.ingestion.builder      import DocumentBuilder
from chatbot.retrieval.vector_store import VectorStoreManager
from langchain_openai               import AzureOpenAIEmbeddings


# ---------------------------------------------------------------------------
# Activity input/output dataclasses
# (Temporal serializes these as JSON automatically)
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing      import Optional


@dataclass
class ParseInput:
    pdf_path:  str
    cache_dir: str


@dataclass
class ParseOutput:
    source_file:     str
    report_date:     str
    collection_date: str
    sections:        dict
    is_valid:        bool
    error:           str = ""


@dataclass
class EmbedInput:
    cache_dir:   str
    persist_dir: str
    force:       bool = False


@dataclass
class EmbedOutput:
    total_documents: int
    skipped:         bool = False


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

@activity.defn(name="parse_pdf")
async def activity_parse_pdf(inp: ParseInput) -> ParseOutput:
    """
    Parse one PDF to structured JSON using Docling.
    Zero LLM tokens.
    """
    logger = activity.logger
    logger.info(f"Parsing: {inp.pdf_path}")

    try:
        parser = PDFParser(cache_dir=Path(inp.cache_dir))
        result = parser.parse(Path(inp.pdf_path))

        total = sum(len(t) for t in result.get("sections", {}).values())
        logger.info(
            f"Done: {Path(inp.pdf_path).name} | "
            f"sections={len(result.get('sections', {}))} | "
            f"tests={total}"
        )

        return ParseOutput(
            source_file     = result.get("source_file", ""),
            report_date     = result.get("report_date", ""),
            collection_date = result.get("collection_date", ""),
            sections        = result.get("sections", {}),
            is_valid        = total > 0,
        )

    except Exception as e:
        logger.error(f"Parse failed: {e}")
        return ParseOutput(
            source_file     = Path(inp.pdf_path).name,
            report_date     = "",
            collection_date = "",
            sections        = {},
            is_valid        = False,
            error           = str(e),
        )


@activity.defn(name="validate_report")
async def activity_validate_report(report: ParseOutput) -> bool:
    """Validate parsed report has required fields."""
    logger = activity.logger
    issues = []

    if not report.source_file:
        issues.append("missing source_file")
    if not report.report_date:
        issues.append("missing report_date")
    if not report.sections:
        issues.append("no sections found")

    total = sum(len(t) for t in report.sections.values())
    if total == 0:
        issues.append("no tests found")

    if issues:
        logger.warning(f"{report.source_file}: {issues}")
        return False

    logger.info(f"{report.source_file}: valid | tests={total}")
    return True


@activity.defn(name="build_documents")
async def activity_build_documents(report: ParseOutput) -> int:
    """
    Convert parsed report to LangChain Documents and cache count.
    Returns document count.
    """
    logger  = activity.logger
    builder = DocumentBuilder()

    report_dict = {
        "source_file":     report.source_file,
        "report_date":     report.report_date,
        "collection_date": report.collection_date,
        "sections":        report.sections,
    }

    docs = builder.build(report_dict)
    logger.info(f"{report.source_file}: built {len(docs)} documents")
    return len(docs)


@activity.defn(name="embed_documents")
async def activity_embed_documents(inp: EmbedInput) -> EmbedOutput:
    """
    Load all cached JSON reports and embed into ChromaDB.
    """
    logger = activity.logger
    config = load_config()

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint   = config["AZURE_OPENAI_ENDPOINT"],
        api_key          = config["AZURE_OPENAI_API_KEY"],
        azure_deployment = config["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
        api_version      = config["AZURE_OPENAI_EMBEDDINGS_API_VERSION"],
    )

    vs = VectorStoreManager(
        embeddings  = embeddings,
        persist_dir = inp.persist_dir,
    )

    if not vs.is_empty() and not inp.force:
        count = vs.count()
        logger.info(f"Chroma already has {count} docs. Skipping.")
        return EmbedOutput(total_documents=count, skipped=True)

    # Load all cached reports
    cache_dir = Path(inp.cache_dir)
    builder   = DocumentBuilder()
    all_docs  = []

    for json_file in sorted(cache_dir.glob("*.json")):
        try:
            report = json.loads(json_file.read_text())
            docs   = builder.build(report)
            all_docs.extend(docs)
            logger.info(f"Built docs from: {json_file.name}")
        except Exception as e:
            logger.error(f"Failed {json_file.name}: {e}")

    if not all_docs:
        logger.error("No documents to embed")
        return EmbedOutput(total_documents=0)

    vs.add(all_docs)
    logger.info(f"Indexed {len(all_docs)} documents")
    return EmbedOutput(total_documents=len(all_docs))