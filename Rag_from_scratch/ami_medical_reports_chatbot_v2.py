from pathlib import Path
import os
import sys
import json
import re

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config


# ---------------------------------------------------------------------------
# Docling converter
# ---------------------------------------------------------------------------

def build_converter() -> DocumentConverter:
    try:
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=True,
            generate_page_images=True,
            generate_table_images=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE,
                do_cell_matching=True,
            ),
        )
    except Exception:
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=True,
        )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_section_name(col) -> str:
    if not isinstance(col, str):
        return ""
    if "." in col:
        return col.split(".")[0].strip()
    return ""


def parse_column_name(col) -> str:
    if not isinstance(col, str):
        return ""
    col = col.split(".")[-1].strip().lower()
    col = re.split(r"\s{2,}|_\d+", col)[0].strip()
    return col


def parse_value(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in {"none", "nan", ""} else s


def extract_date_from_col(col) -> str:
    if not isinstance(col, str):
        return ""
    match = re.search(
        r"(\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{4}|\d{1,2}/\d{1,2}/\d{4})",
        col
    )
    return match.group(1).strip() if match else ""


def extract_all_dates(markdown: str) -> dict:
    """
    Extract dates from markdown text.
    Handles both same-line and next-line date formats.
    Same line:  "Reporting DateTime: 30-Apr-2026 14:23"
    Next line:  "Ordered On.......\n16/05/2026 00:00"
    """
    dates = {
        "report_date":      "",
        "collection_date":  "",
        "ordered_date":     "",
    }

    # Date value pattern
    # Curly braces doubled {{ }} so .format() does not treat them as placeholders
    DATE_VALUE = (
        r"(\d{{1,2}}[-/][A-Za-z]{{3,9}}[-/]\d{{4}}"   # 30-Apr-2026
        r"|\d{{1,2}}/\d{{1,2}}/\d{{4}}"                # 16/05/2026
        r"|\d{{4}}-\d{{2}}-\d{{2}})"                   # 2026-05-16
    )

    # Label followed by date on SAME line or NEXT line
    LABEL_THEN_DATE = r"{label}[\s.:_-]*\n?\s*" + DATE_VALUE

    patterns = {
        "report_date": [
            LABEL_THEN_DATE.format(label=r"Reporting\s*Date(?:Time)?"),
            LABEL_THEN_DATE.format(label=r"Verified\s*On"),
            LABEL_THEN_DATE.format(label=r"Report\s*Date"),
            LABEL_THEN_DATE.format(label=r"Reported"),
        ],
        "collection_date": [
            LABEL_THEN_DATE.format(label=r"Collection\s*Date(?:Time)?"),
            LABEL_THEN_DATE.format(label=r"Received\s*in\s*Lab"),
            LABEL_THEN_DATE.format(label=r"Collected"),
            LABEL_THEN_DATE.format(label=r"Sample\s*(?:Collected|Received)"),
        ],
        "ordered_date": [
            LABEL_THEN_DATE.format(label=r"Ordered\s*On"),
            LABEL_THEN_DATE.format(label=r"Order\s*Date"),
            LABEL_THEN_DATE.format(label=r"Registration\s*Date"),
        ],
    }

    # Standalone DATE_VALUE for fallback searches (no .format() used here)
    DATE_VALUE_RAW = (
        r"(\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{4}"
        r"|\d{1,2}/\d{1,2}/\d{4}"
        r"|\d{4}-\d{2}-\d{2})"
    )

    for field, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                dates[field] = match.group(1).strip()
                break

    # Fallback: use ordered or collection date as report date
    if not dates["report_date"]:
        dates["report_date"] = (
            dates["ordered_date"] or
            dates["collection_date"]
        )

    # Last resort: first date found anywhere in document
    if not dates["report_date"]:
        match = re.search(DATE_VALUE_RAW, markdown)
        if match:
            dates["report_date"] = match.group(1).strip()

    return dates

def compute_flag(value: str, reference_range: str) -> tuple[str, bool]:
    """
    Pure math, zero LLM tokens.
    Returns (flag, is_abnormal).
    Skips non-numeric values like Nil, Negative, Normal.
    """
    match = re.search(r"([\d.]+)\s*-\s*([\d.]+)", str(reference_range))
    if not match or not value:
        return "", False
    try:
        val  = float(str(value).replace(",", ""))
        low  = float(match.group(1))
        high = float(match.group(2))
        if val < low:
            return "low", True
        elif val > high:
            return "high", True
        return "", False
    except (ValueError, TypeError):
        return "", False


def detect_col_roles(cols: list) -> dict:
    """
    Detect column roles from headers.
    Handles both named string columns and integer columns.
    """
    roles = {
        "test":  [],
        "value": [],
        "unit":  [],
        "range": [],
        "date":  "",
    }

    # Named string columns
    if all(isinstance(c, str) for c in cols):
        for i, col in enumerate(cols):
            clean = parse_column_name(col)
            if clean in {"test", "parameter", "parameter name", "investigation", "analyte"}:
                roles["test"].append(i)
            elif "result" in clean or "value" in clean or "finding" in clean:
                roles["value"].append(i)
                if not roles["date"]:
                    roles["date"] = extract_date_from_col(col)
            elif clean in {"unit", "units"}:
                roles["unit"].append(i)
            elif any(k in clean for k in ["range", "normal", "ref", "interval"]):
                roles["range"].append(i)
        return roles

    # Integer columns: assume pattern [param, result, ref, unit, param, result, ref, unit]
    n = len(cols)
    if n >= 4:
        roles["test"].append(0)
        roles["value"].append(1)
        roles["range"].append(2)
        if n > 3:
            roles["unit"].append(3)
        if n >= 8:
            roles["test"].append(4)
            roles["value"].append(5)
            roles["range"].append(6)
            roles["unit"].append(7)
        elif n >= 6:
            roles["test"].append(4)
            roles["value"].append(5)

    return roles


def table_to_tests(df, section_hint: str = "") -> tuple[str, list[dict], str]:
    """
    Returns (section_name, tests, report_date).
    Handles:
    - Named columns: 'Renal Function Test.Test', 'Normal Range' x2
    - Integer columns: 0,1,2,3,4,5,6,7 (two side-by-side groups)
    """
    cols = list(df.columns)

    section_name = ""
    if isinstance(cols[0], str):
        section_name = parse_section_name(cols[0])
    section_name = section_name or section_hint or "General Lab Tests"

    roles       = detect_col_roles(cols)
    report_date = roles["date"]

    HEADER_KEYWORDS = {
        "parameter name", "parameter", "test", "result",
        "reference value", "reference", "unit", "units",
        "chemical examination", "physical examination",
        "microscopic examination", "colour", "turbidity",
        "deposit",
    }

    tests = []
    for _, row in df.iterrows():
        values = list(row)

        first_val = parse_value(values[0]).lower()
        if first_val in HEADER_KEYWORDS or first_val.endswith("examination"):
            continue

        for group_idx, t_col in enumerate(roles["test"]):
            v_col = roles["value"][group_idx] if group_idx < len(roles["value"]) else None
            u_col = roles["unit"][group_idx]  if group_idx < len(roles["unit"])  else None

            if roles["range"]:
                if all(isinstance(c, str) for c in cols):
                    r_cols = roles["range"]
                else:
                    r_cols = [roles["range"][group_idx]] if group_idx < len(roles["range"]) else []
            else:
                r_cols = []

            test_name = parse_value(values[t_col]) if t_col < len(values) else ""
            value     = parse_value(values[v_col]) if v_col is not None and v_col < len(values) else ""
            unit      = parse_value(values[u_col]) if u_col is not None and u_col < len(values) else ""

            range_parts     = [parse_value(values[ri]) for ri in r_cols if ri < len(values)]
            range_parts     = [p for p in range_parts if p]
            reference_range = " ".join(range_parts)
            reference_range = re.sub(r"\s*-\s*-\s*", " - ", reference_range)
            reference_range = re.sub(r"\s+", " ", reference_range).strip()

            if not test_name or test_name.lower() in HEADER_KEYWORDS:
                continue

            flag, is_abnormal = compute_flag(value, reference_range)

            tests.append({
                "test_name":       test_name,
                "value":           value,
                "unit":            unit,
                "reference_range": reference_range,
                "flag":            flag,
                "is_abnormal":     is_abnormal,
            })

    return section_name, tests, report_date


# ---------------------------------------------------------------------------
# Docling extraction -- zero LLM tokens
# ---------------------------------------------------------------------------

def docling_extract(pdf_path: Path, converter: DocumentConverter) -> dict:
    """
    Extract everything from PDF using Docling only.
    Zero LLM calls. Zero tokens used.
    """
    doc      = converter.convert(str(pdf_path)).document
    markdown = doc.export_to_markdown()

    print(f"\n--- Docling Markdown Start: {pdf_path.name} ---\n")
    print(markdown)
    print(f"\n--- Docling Markdown End: {pdf_path.name} ---\n")

    # Extract all dates from markdown text
    dates = extract_all_dates(markdown)

    output = {
        "report_date":       dates["report_date"],
        "collection_date":   dates["collection_date"],
        "source_file":       pdf_path.name,
        "sections":          {},
    }

    # Get section headings from markdown for integer-column tables
    section_headings = re.findall(r"^#{1,3}\s+(.+)$", markdown, re.MULTILINE)
    section_headings = [
        h.strip() for h in section_headings
        if not any(skip in h.lower() for skip in ["image", "page", "www", "dr.", "dr "])
    ]

    for table_idx, table in enumerate(doc.tables):
        try:
            df = table.export_to_dataframe(doc)
        except Exception:
            continue

        if df is None or df.empty:
            continue

        section_hint = section_headings[table_idx] if table_idx < len(section_headings) else ""
        section_name, tests, col_date = table_to_tests(df, section_hint)

        # Use column date only if markdown date not found
        if col_date and not output["report_date"]:
            output["report_date"] = col_date

        if tests:
            if section_name in output["sections"]:
                section_name = f"{section_name} ({table_idx})"
            output["sections"][section_name] = tests

    return output


# ---------------------------------------------------------------------------
# Cache -- never reprocess same PDF
# ---------------------------------------------------------------------------

def get_or_process(
    pdf_path:  Path,
    converter: DocumentConverter,
    cache_dir: Path = Path("./json_cache"),
) -> dict:
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{pdf_path.stem}.json"

    if cache_file.exists():
        print(f"Cache hit -> {pdf_path.name}")
        return json.loads(cache_file.read_text())

    print(f"Docling -> {pdf_path.name}")
    result = docling_extract(pdf_path, converter)

    cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Cached  -> {cache_file}")
    return result


# ---------------------------------------------------------------------------
# Convert JSON -> LangChain Documents
# ---------------------------------------------------------------------------

def report_to_documents(report: dict) -> list[Document]:
    documents        = []
    report_date      = report.get("report_date",      "Unknown")
    collection_date  = report.get("collection_date",  "Unknown")
    source_file      = report.get("source_file",      "unknown")

    all_tests      = []
    abnormal_tests = []

    for section_name, tests in report.get("sections", {}).items():
        for test in tests:
            is_abnormal  = test.get("is_abnormal", False)
            flag         = test.get("flag", "")
            abnormal_tag = f" [ABNORMAL - {flag.upper()}]" if is_abnormal else ""

            documents.append(Document(
                page_content=(
                    f"Test: {test['test_name']}{abnormal_tag}\n"
                    f"Result: {test['value']} {test['unit']}\n"
                    f"Reference Range: {test['reference_range'] or 'N/A'}\n"
                    f"Section: {section_name}\n"
                    f"Report Date: {report_date}\n"
                    f"Collection Date: {collection_date}\n"
                    f"Source: {source_file}"
                ),
                metadata={
                    "doc_type":        "individual_test",
                    "test_name":       test["test_name"].lower(),
                    "value":           test["value"],
                    "unit":            test["unit"],
                    "reference_range": test["reference_range"],
                    "is_abnormal":     is_abnormal,
                    "flag":            flag,
                    "section":         section_name,
                    "report_date":     report_date,
                    "collection_date": collection_date,
                    "source_file":     source_file,
                }
            ))

            all_tests.append(test)
            if is_abnormal:
                abnormal_tests.append((section_name, test))

    # Summary document
    all_text = "\n".join(
        f"  {t['test_name']}: {t['value']} {t['unit']}"
        + (f"  ABNORMAL {t['flag'].upper()}" if t["is_abnormal"] else "")
        for t in all_tests
    )
    abnormal_text = "\n".join(
        f"  [{s}] {t['test_name']}: {t['value']} {t['unit']} "
        f"(Range: {t['reference_range']}) {t['flag'].upper()}"
        for s, t in abnormal_tests
    ) or "  None"

    documents.append(Document(
        page_content=(
            f"Lab Report Summary\n"
            f"Report Date: {report_date}\n"
            f"Collection Date: {collection_date}\n"
            f"Source: {source_file}\n"
            f"Total Tests: {len(all_tests)}\n"
            f"Abnormal Count: {len(abnormal_tests)}\n\n"
            f"All Results:\n{all_text}\n\n"
            f"Abnormal Results:\n{abnormal_text}"
        ),
        metadata={
            "doc_type":       "report_summary",
            "report_date":    report_date,
            "collection_date": collection_date,
            "source_file":    source_file,
            "abnormal_count": len(abnormal_tests),
            "total_tests":    len(all_tests),
        }
    ))

    return documents


# ---------------------------------------------------------------------------
# Smart retriever
# ---------------------------------------------------------------------------

def smart_retriever(vector_store, question: str):
    q = question.lower()

    if any(w in q for w in ["abnormal", "high", "low", "flag", "critical", "concern"]):
        return vector_store.as_retriever(
            search_kwargs={"k": 10, "filter": {"is_abnormal": True}}
        )
    if any(w in q for w in ["summary", "all tests", "overview", "full report"]):
        return vector_store.as_retriever(
            search_kwargs={"k": 5, "filter": {"doc_type": "report_summary"}}
        )
    if any(w in q for w in ["when", "date", "performed", "which month", "collected"]):
        return vector_store.as_retriever(
            search_kwargs={"k": 5, "filter": {"doc_type": "report_summary"}}
        )
    if any(w in q for w in ["improve", "better", "worse", "trend", "compare", "progress"]):
        return vector_store.as_retriever(search_kwargs={"k": 20})

    return vector_store.as_retriever(
        search_kwargs={"k": 6, "filter": {"doc_type": "individual_test"}}
    )


def format_docs(documents):
    parts = []
    for doc in documents:
        source     = doc.metadata.get("source_file",    "unknown")
        rep_date   = doc.metadata.get("report_date",    "?")
        coll_date  = doc.metadata.get("collection_date","?")
        parts.append(
            f"[Source: {source} | Report Date: {rep_date} | Collection Date: {coll_date}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def debug_print_chunks(question: str, documents: list[Document]):
    print("\n" + "=" * 80)
    print("DEBUG: CHUNKS SENT TO LLM")
    print(f"Question: {question}")
    print(f"Total chunks: {len(documents)}")
    print("=" * 80)
    for index, doc in enumerate(documents, start=1):
        source    = doc.metadata.get("source_file",    "unknown")
        rep_date  = doc.metadata.get("report_date",    "?")
        coll_date = doc.metadata.get("collection_date","?")
        doc_type  = doc.metadata.get("doc_type",       "unknown")
        print(f"\n--- Chunk {index}/{len(documents)} ---")
        print(f"source={source} | report_date={rep_date} | collection_date={coll_date} | type={doc_type}")
        print(doc.page_content)
    print("\n" + "=" * 80)


def retrieve_context(vector_store, question: str) -> str:
    retriever = smart_retriever(vector_store, question)
    documents = retriever.invoke(question)
    debug_print_chunks(question, documents)
    return format_docs(documents)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

MEDICAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical records assistant helping review lab reports.

Guidelines:
- Answer ONLY from the provided context
- If asked whether a test was performed: YES or NO, cite source and date
- If asked for a result: quote exact value and reference range
- If asked for a summary: list every test, result, unit, reference range
- If asked about abnormal results: flag values outside range, label ABNORMAL
- If asked when a test was performed: extract collection date or report date from context
- If asked about improvements: compare values across reports by date,
  state if better, worse or same with actual numbers
- If not found: say Not found in the provided lab reports
- NEVER diagnose. NEVER prescribe. Say consult your doctor for abnormals"""),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def trim_to_last_4(session_id: str):
    history = get_session_history(session_id)
    if len(history.messages) > 4:
        history.messages = history.messages[-4:]


# ---------------------------------------------------------------------------
# Build RAG chain
# ---------------------------------------------------------------------------

def build_rag_chain(source_path: Path):
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

    vector_store = Chroma(
        persist_directory="./chroma_medical_db",
        embedding_function=embeddings,
    )

    if vector_store._collection.count() == 0:
        converter     = build_converter()
        pdf_files     = (
            [source_path] if source_path.is_file()
            else sorted(source_path.rglob("*.pdf"))
        )
        all_documents = []
        for pdf_file in pdf_files:
            report = get_or_process(pdf_file, converter)
            docs   = report_to_documents(report)
            all_documents.extend(docs)
            print(f"  {len(docs)} docs from {pdf_file.name}")

        vector_store.add_documents(all_documents)
        print(f"Indexed {len(all_documents)} documents total")
    else:
        print("Reusing existing Chroma store")

    core_chain = (
        {
            "context": RunnableLambda(
                lambda x: retrieve_context(vector_store, x["question"])
            ),
            "question":     RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | MEDICAL_PROMPT
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pdf_path   = Path("/home/abbas/Downloads/Ami_fever_2026-20260612T142009Z-3-001/all_pdfs/")
    session_id = "session_1"

    chain = build_rag_chain(pdf_path)

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

        trim_to_last_4(session_id)

        answer = chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )

        print(f"\nAnswer:\n{answer}\n")