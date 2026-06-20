import pdfplumber
import fitz  # PyMuPDF
import re
import json


def extract_text_with_pymupdf(pdf_path):
    """Fallback full-text extraction"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text


def extract_tables_with_pdfplumber(pdf_path):
    """Extract tables (best for lab reports)"""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append(table)
    return tables


def normalize_row(row):
    """Clean row values"""
    return [cell.strip() if cell else "" for cell in row]


def parse_lab_table(table):
    """
    Parse a table into structured lab tests
    Assumes columns like:
    Test | Value | Unit | Reference Range
    """
    structured_tests = []

    headers = normalize_row(table[0])

    for row in table[1:]:
        row = normalize_row(row)

        if len(row) < 2:
            continue

        test_data = {}

        for i, cell in enumerate(row):
            col = headers[i].lower()

            if "test" in col or "parameter" in col:
                test_data["test_name"] = cell
            elif "result" in col or "value" in col:
                test_data["value"] = cell
            elif "unit" in col:
                test_data["unit"] = cell
            elif "ref" in col or "range" in col:
                test_data["reference_range"] = cell
            elif "flag" in col:
                test_data["flag"] = cell

        if test_data:
            structured_tests.append(test_data)

    return structured_tests


def extract_patient_info(text):
    """Basic regex extraction for patient info"""
    patient_info = {}

    patterns = {
        "name": r"Name[:\s]+([A-Za-z ]+)",
        "age": r"Age[:\s]+(\d+)",
        "gender": r"Gender[:\s]+(Male|Female)",
        "report_date": r"Date[:\s]+([\d/-]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            patient_info[key] = match.group(1)

    return patient_info


def classify_sections(tables):
    """
    Try grouping tables into sections (basic heuristic)
    """
    sections = []

    for idx, table in enumerate(tables):
        parsed_tests = parse_lab_table(table)

        if not parsed_tests:
            continue

        section = {
            "section_name": f"Section_{idx+1}",
            "tests": parsed_tests
        }

        sections.append(section)

    return sections


def pdf_to_structured_json(pdf_path):
    # Step 1: Extract text + tables
    print("Extracting text...")
    raw_text = extract_text_with_pymupdf(pdf_path)

    print("Extracting tables...")
    tables = extract_tables_with_pdfplumber(pdf_path)

    # Step 2: Extract structured components
    print("Parsing patient info...")
    patient_info = extract_patient_info(raw_text)

    print("Parsing lab sections...")
    sections = classify_sections(tables)

    # Step 3: Combine everything
    final_json = {
        "patient_info": patient_info,
        "sections": sections,
        "raw_text": raw_text[:1000]  # optional preview
    }

    return final_json


# ✅ Run script
if __name__ == "__main__":
    pdf_path = "/home/abbas/Documents/IMG_20260613_154648.pdf"

    result = pdf_to_structured_json(pdf_path)

    # Save JSON
    with open("lab_report.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\n✅ Done! Output saved to lab_report.json")