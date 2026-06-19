from pathlib import Path
import os
import sys
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config


# ---------------------------------------------------------------------------
# Load PDF (same mechanism as your code)
# ---------------------------------------------------------------------------
def load_pdf_text(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # Combine all pages into one text blob
    full_text = "\n\n".join([doc.page_content for doc in documents])
    return full_text


# ---------------------------------------------------------------------------
# Prompt (structured extraction)
# ---------------------------------------------------------------------------
MEDICAL_JSON_PROMPT = ChatPromptTemplate.from_template(
    """You are a medical data extractor.

Convert the lab report below into structured JSON.

Requirements:
- Extract patient_info:
  - name
  - age
  - gender
  - report_date
- Extract all tests
- For each test include:
  - test_name
  - value
  - unit
  - reference_range (if present)
  - flag (high/low if mentioned)
- Group tests into sections if possible (e.g., CBC, LFT)

Return ONLY valid JSON. No explanation.

Lab Report:
{text}
"""
)


# ---------------------------------------------------------------------------
# Build simple chain (NO RAG)
# ---------------------------------------------------------------------------
def build_chain():
    config = load_config()

    llm = AzureChatOpenAI(
        azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"],
        azure_deployment=config["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=config["AZURE_OPENAI_CHAT_API_VERSION"],
        temperature=0,
    )

    chain = MEDICAL_JSON_PROMPT | llm | StrOutputParser()
    return chain


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pdf_path = "/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images/failed_downloads/pdfs/IMG_20260613_161253.pdf"
    pdf_path = "/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images/downloaded_pdfs/manually_downloaded/20260614103618570_6c38c284-a683-40b6-9c34-c9bfec06d15b.pdf"

    print("Loading PDF...")
    raw_text = load_pdf_text(pdf_path)

    # Optional safety (token limit)
    raw_text = raw_text[:15000]

    print("Sending to LLM...")
    chain = build_chain()

    result = chain.invoke({"text": raw_text})

    # Save JSON
    with open("structured_output.json", "w") as f:
        try:
            json.dump(json.loads(result), f, indent=2)
        except:
            f.write(result)

    print("✅ Done! Output saved to structured_output.json")