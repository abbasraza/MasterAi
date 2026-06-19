import json
from pathlib import Path
import re

from docling.document_converter import DocumentConverter


# ---------------------------------------------------------------------------
# Load and convert PDF
# ---------------------------------------------------------------------------
def load_document(pdf_path: Path):
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document


# ---------------------------------------------------------------------------
# Extract structured lab data
# ---------------------------------------------------------------------------
def extract_lab_data(doc):
    output = {
        "patient_info": {},
        "sections": []
    }

    # -----------------------------------------------------------------------
    # 1. Extract tables (fixed)
    # -----------------------------------------------------------------------
    for i, table in enumerate(doc.tables):

        # FIX 1: correct API usage (pass doc)
        try:
            df = table.export_to_dataframe(doc)

            if df is None or df.empty:
                continue

            headers = [str(col).lower() for col in df.columns]
            rows = df.values.tolist()

        except Exception:
            # fallback for compatibility
            table_data = table.export_to_otsl()

            if not table_data or len(table_data) < 2:
                continue

            headers = [str(h).lower() for h in table_data[0]]
            rows = table_data[1:]

        tests = []

        for row in rows:
            # skip empty rows
            if not any(row):
                continue

            test = {
                "test_name": None,
                "value": None,
                "unit": None,
                "reference_range": None,
                "flag": None
            }

            for j, cell in enumerate(row):
                if j >= len(headers):
                    continue

                col = headers[j]

                if "test" in col or "parameter" in col:
                    test["test_name"] = str(cell)
                elif "result" in col or "value" in col:
                    test["value"] = str(cell)
                elif "unit" in col:
                    test["unit"] = str(cell)
                elif "ref" in col or "range" in col:
                    test["reference_range"] = str(cell)
                elif "flag" in col:
                    test["flag"] = str(cell)

            if test["test_name"]:
                tests.append(test)

        if tests:
            output["sections"].append({
                "section_name": f"Table_{i}",
                "tests": tests
            })

    # -----------------------------------------------------------------------
    # 2. Extract patient info from raw text
    # -----------------------------------------------------------------------
    full_text = doc.export_to_markdown()

    patterns = {
        "name": r"Name[:\s]+([A-Za-z ]+)",
        "age": r"Age[:\s]+(\d+)",
        "gender": r"Gender[:\s]+(Male|Female)",
        "report_date": r"(Date|Collected)[:\s]+([\d/-]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            output["patient_info"][key] = match.group(1 if key != "report_date" else 2)

    return output


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pdf_path = Path("/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images/downloaded_pdfs/manually_downloaded/20260614103618570_6c38c284-a683-40b6-9c34-c9bfec06d15b.pdf")

    print("Parsing PDF with Docling...")
    doc = load_document(pdf_path)

    print("Extracting structured data...")
    structured_json = extract_lab_data(doc)

    with open("structured_output.json", "w") as f:
        json.dump(structured_json, f, indent=2)

    print("Done! Saved to structured_output.json")