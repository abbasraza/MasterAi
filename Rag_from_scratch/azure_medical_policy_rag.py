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


def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)


def load_documents(source_path: Path):
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
            documents.extend(PyPDFLoader(str(pdf_file)).load())
        return documents, pdf_files

    raise ValueError(f"Path does not exist: {source_path}")


def build_rag_chain(source_path: Path):
    config = load_config()
    persist_directory = "./chroma_hr_db"

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
    pdf_files = []
    documents = []
    chunks = []

    if existing_count == 0:
        documents, pdf_files = load_documents(source_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        vector_store.add_documents(documents=chunks)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = AzureChatOpenAI(
        azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"],
        azure_deployment=config["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=config["AZURE_OPENAI_CHAT_API_VERSION"],
        temperature=.2,
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, len(pdf_files), len(documents), len(chunks), existing_count == 0


if __name__ == "__main__":
    pdf_path = Path("/home/abbas/Downloads/PK Policies for AI Implementation/")

    chain, pdf_count, page_count, chunk_count, indexed_now = build_rag_chain(pdf_path)
    if indexed_now:
        print(f"Indexed {pdf_count} PDF files, {page_count} pages, and created {chunk_count} chunks.")
    else:
        print("Reused existing persisted Chroma store from ./chroma_hr_db (no re-embedding).")

    print("\nAsk questions about the policy. Type 'exit' or 'quit' to stop.\n")
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
        print("\nAnswer:\n")
        print(answer)
        print()