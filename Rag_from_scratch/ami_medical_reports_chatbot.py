from pathlib import Path
import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_docs(documents):
    """Concatenate page content AND inject source metadata so the LLM
    can reference which file / page a result came from."""
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown file")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Source: {Path(source).name}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def load_documents(source_path: Path):
    """Load one PDF or every PDF inside a directory."""
    if source_path.is_file():
        if source_path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF: {source_path}")
        return PyPDFLoader(str(source_path)).load(), [source_path]

    if source_path.is_dir():
        pdf_files = sorted(source_path.rglob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {source_path}")
        documents = []
        for pdf_file in pdf_files:
            docs = PyPDFLoader(str(pdf_file)).load()
            # ----------------------------------------------------------------
            # Enrich metadata with the filename so retrieval context carries
            # the lab-report name (useful for "when was this test done?")
            # ----------------------------------------------------------------
            for doc in docs:
                doc.metadata["source"] = str(pdf_file)
            documents.extend(docs)
        return documents, pdf_files

    raise ValueError(f"Path does not exist: {source_path}")


# ---------------------------------------------------------------------------
# Medical-specific prompt
# ---------------------------------------------------------------------------

MEDICAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a medical records assistant. Use ONLY the lab report context below to answer.

Guidelines:
- If asked whether a test was performed: state YES or NO and cite the source file and date if available.
- If asked for a result: quote the exact value and its reference range (normal range) if present.
- If asked for a summary: list every test name, its result, unit, and reference range.
- If asked about abnormal results: flag any value marked HIGH, LOW, *, H, L, or outside the
  reference range. Clearly label them ABNORMAL and explain briefly what the test measures.
- If asked when a test was performed: extract the collection/report date from the context.
- If the information is not in the context, say "Not found in the provided lab reports."
- Never guess or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""
)


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_rag_chain(source_path: Path):
    config = load_config()

    # ── Separate DB from the HR chatbot ────────────────────────────────────
    persist_directory = "./chroma_medical_db"

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"],
        azure_deployment=config["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
        api_version=config["AZURE_OPENAI_EMBEDDINGS_API_VERSION"],
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    existing_count = vector_store._collection.count()
    pdf_files, documents, chunks = [], [], []

    if existing_count == 0:
        documents, pdf_files = load_documents(source_path)

        # ── Smaller chunks: lab tables are compact, dense text ──────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=0,
        )
        chunks = splitter.split_documents(documents)
        vector_store.add_documents(documents=chunks)

    # ── Retrieve more chunks: a single report may span many pages ───────────
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})  # ← was 4

    llm = AzureChatOpenAI(
        azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"],
        azure_deployment=config["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=config["AZURE_OPENAI_CHAT_API_VERSION"],
        temperature=0,   # ← 0 for factual / clinical accuracy (was 0.2)
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | MEDICAL_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, len(pdf_files), len(documents), len(chunks), existing_count == 0


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Point this at your lab-report PDFs ──────────────────────────────────
    pdf_path = Path("/home/abbas/Downloads/Ami_fever_2026-20260612T142009Z-3-001/Ami_fever_2026")

    chain, pdf_count, page_count, chunk_count, indexed_now = build_rag_chain(pdf_path)

    if indexed_now:
        print(f"Indexed {pdf_count} PDF(s), {page_count} pages → {chunk_count} chunks.")
    else:
        print("Reused existing Chroma store ./chroma_medical_db (no re-embedding).")

    print("\n Medical Records Assistant")
    print("Ask about tests, results, dates, or abnormal findings.")
    print("Type 'exit' or 'quit' to stop.\n")

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

        answer = chain.invoke(question)
        print(f"\nAnswer:\n{answer}\n")