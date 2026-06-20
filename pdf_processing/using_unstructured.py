#!/usr/bin/env python3
"""
Generic lab report parser using Unstructured.
Works for any single-value chart report (ESR, HbA1c, CRP etc.)
Not coupled to any specific test.

Usage: python parse_lab_report.py <pdf_path>
"""

import json
import re
import sys
from pathlib import Path


# =============================================================================
# Install:
# pip install "unstructured[pdf]" pdfminer.six
# =============================================================================


# =============================================================================
# Known test catalog (extend as needed)
# =============================================================================

TEST_CATALOG = {
    # keyword       full name                          unit          section
    "esr":         ("ESR (Erythrocyte Sedimentation Rate)", "mm/1st Hr", "Hematology"),
    "hba1c":       ("HbA1c (Glycated Hemoglobin)",          "%",          "Chemical Pathology"),
    "crp":         ("C-Reactive Protein (CRP)",              "mg/L",       "Immunology"),
    "glucose":     ("Blood Glucose",                         "mg/dL",      "Chemical Pathology"),
    "cholesterol": ("Total Cholesterol",                     "mg/dL",      "Chemical Pathology"),
    "tsh":         ("Thyroid Stimulating Hormone (TSH)",     "uIU/mL",     "Endocrinology"),
    "vitamin d":   ("Vitamin D (25-OH)",                     "ng/mL",      "Chemical Pathology"),
    "vitamin b12": ("Vitamin B12",                           "pg/mL",      "Chemical Pathology"),
    "ferritin":    ("Ferritin",                              "ng/mL",      "Chemical Pathology"),
    "uric acid":   ("Uric Acid",                             "mg/dL",      "Chemical Pathology"),
}

# Unit patterns to detect result line generically
UNIT_PATTERNS = [
    r"mm/1st\s*[Hh](?:r|our)?",   # ESR
    r"mm/hr",
    r"(?<!\w)%(?!\w)",             # percentage
    r"mg/dL",
    r"mg/L",
    r"g/dL",
    r"g/L",
    r"U/L",
    r"uIU/mL",
    r"mIU/mL",
    r"ng/mL",
    r"pg/mL",
    r"nmol/L",
    r"mmol/L",
    r"µmol/L",
    r"umol/L",
    r"IU/L",
    r"x10\^?\d+/[lL]",            # CBC counts
]

COMBINED_UNIT_RE = re.compile(
    "|".join(f"({p})" for p in UNIT_PATTERNS),
    re.IGNORECASE,
)


# =============================================================================
# Step 1: Extract text using Unstructured
# =============================================================================

def extract_text(pdf_path: str) -> tuple[list, str]:
    """
    Extract elements and full text using Unstructured.
    Tries strategies: fast -> auto -> hi_res
    """
    from unstructured.partition.pdf import partition_pdf

    strategies = ["fast", "auto"]
    elements   = None

    for strategy in strategies:
        try:
            print(f"  Trying strategy: {strategy}")
            elements = partition_pdf(
                filename              = pdf_path,
                strategy              = strategy,
                infer_table_structure = True,
                include_metadata      = True,
            )
            if elements:
                print(f"  Success: {len(elements)} elements")
                break
        except Exception as e:
            print(f"  Failed ({strategy}): {e}")

    if not elements:
        raise RuntimeError("All Unstructured strategies failed")

    full_text = "\n".join(
        el.text.strip()
        for el in elements
        if el.text and el.text.strip()
    )

    return elements, full_text


# =============================================================================
# Step 2: Detect which tests are in this report
# =============================================================================

def detect_tests(full_text: str) -> list[dict]:
    """
    Detect which tests appear in the report.
    Returns list of detected test info dicts.
    """
    text_lower  = full_text.lower()
    found_tests = []

    for keyword, (full_name, default_unit, section) in TEST_CATALOG.items():
        if keyword in text_lower:
            found_tests.append({
                "keyword":  keyword,
                "name":     full_name,
                "unit":     default_unit,
                "section":  section,
            })
            print(f"  Detected test: {full_name}")

    # If nothing found in catalog, try to detect from unit patterns
    if not found_tests:
        unit_match = COMBINED_UNIT_RE.search(full_text)
        if unit_match:
            found_tests.append({
                "keyword": "unknown",
                "name":    "Unknown Test",
                "unit":    unit_match.group().strip(),
                "section": "General Lab Tests",
            })

    return found_tests


# =============================================================================
# Step 3: Generic value extraction (not test-specific)
# =============================================================================

def extract_value(full_text: str, test_keyword: str) -> tuple[str, str]:
    """
    Generically extract value and unit for any test.
    Returns (value, unit).

    Strategy A: number + unit on same line  e.g. "41mm/1st H"
    Strategy B: number on line after test name
    Strategy C: largest standalone number after test heading
    """
    lines = full_text.splitlines()

    # ── Strategy A: number directly attached to unit ──────────────────────
    # e.g. "41mm/1st H", "6.5%", "120mg/dL"
    combined = re.search(
        r"(\d+(?:\.\d+)?)\s*(" + "|".join(UNIT_PATTERNS) + r")",
        full_text, re.IGNORECASE
    )
    if combined:
        return combined.group(1), combined.group(2).strip()

    # ── Strategy B: number on line immediately after test heading ──────────
    for i, line in enumerate(lines):
        if test_keyword.lower() in line.lower():
            for next_line in lines[i+1:i+6]:
                next_line = next_line.strip()

                # Number with unit on same line
                num_unit = re.match(
                    r"^(\d+(?:\.\d+)?)\s*(" + "|".join(UNIT_PATTERNS) + r")",
                    next_line, re.IGNORECASE
                )
                if num_unit:
                    return num_unit.group(1), num_unit.group(2).strip()

                # Standalone number (no unit yet)
                num_only = re.match(r"^(\d+(?:\.\d+)?)\s*$", next_line)
                if num_only:
                    return num_only.group(1), ""
            break

    # ── Strategy C: find all numbers, pick most prominent ─────────────────
    # Filter out years, phone numbers, specimen IDs
    def is_result_candidate(n: str) -> bool:
        val = float(n)
        if 1900 <= val <= 2100:   return False   # year
        if val > 9999:            return False   # phone/ID
        return True

    all_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", full_text)
    candidates  = [n for n in all_numbers if is_result_candidate(n)]

    if candidates:
        return candidates[0], ""

    return None, ""


# =============================================================================
# Step 4: Generic reference range extraction
# =============================================================================

def extract_reference_range(full_text: str) -> tuple[str, str]:
    """
    Extract reference range from any lab report text.
    Returns (reference_range, flag_hint)

    Handles:
      "Normal (=15)"         -> "0 - 15"
      "Normal (<=15)"        -> "0 - 15"
      "Normal (<15)"         -> "0 - 15"
      "High (>15)"           -> upper bound = 15
      "Less Than 1.0"        -> "0 - 1.0"
      "10 - 45"              -> "10 - 45"
      "Reference: 0 - 15"    -> "0 - 15"
    """
    patterns = [
        # "Normal (=15)" or "Normal (<=15)"
        (r"[Nn]ormal\s*\([=<]{1,2}\s*(\d+(?:\.\d+)?)\)", "upper_only"),
        # "Normal (<15)"
        (r"[Nn]ormal\s*\(<\s*(\d+(?:\.\d+)?)\)",         "upper_only"),
        # "High (>15)"
        (r"[Hh]igh\s*\(>\s*(\d+(?:\.\d+)?)\)",           "upper_only"),
        # "Less Than X"
        (r"[Ll]ess\s+[Tt]han\s+(\d+(?:\.\d+)?)",         "upper_only"),
        # "Greater Than X"
        (r"[Gg]reater\s+[Tt]han\s+(\d+(?:\.\d+)?)",      "lower_only"),
        # Standard "10 - 45" or "10-45"
        (r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)",    "range"),
    ]

    for pattern, kind in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            if kind == "range":
                low  = match.group(1)
                high = match.group(2)
                # Sanity check: low < high
                if float(low) < float(high):
                    return f"{low} - {high}", ""
            elif kind == "upper_only":
                return f"0 - {match.group(1)}", ""
            elif kind == "lower_only":
                return f"{match.group(1)} - 999", ""

    return "", ""


# =============================================================================
# Step 5: Generic flag detection
# =============================================================================

def detect_flag(
    full_text:       str,
    value:           str,
    reference_range: str,
) -> tuple[str, bool]:
    """
    Detect abnormal flag.
    Uses text keywords + numeric comparison.
    """
    flag        = None
    is_abnormal = False

    # Text-based detection
    if re.search(r"\bHigh\b",     full_text, re.IGNORECASE): flag = "high"
    if re.search(r"\bLow\b",      full_text, re.IGNORECASE): flag = "low"
    if re.search(r"\bAbnormal\b", full_text, re.IGNORECASE): flag = "abnormal"
    if re.search(r"\bCritical\b", full_text, re.IGNORECASE): flag = "critical"

    # Numeric validation (overrides text if contradictory)
    if value and reference_range:
        range_match = re.search(
            r"([\d.]+)\s*[-–]\s*([\d.]+)",
            reference_range
        )
        if range_match:
            try:
                val  = float(str(value).replace(",", ""))
                low  = float(range_match.group(1))
                high = float(range_match.group(2))
                if val > high:
                    flag        = "high"
                    is_abnormal = True
                elif val < low:
                    flag        = "low"
                    is_abnormal = True
                else:
                    flag        = None
                    is_abnormal = False
            except (ValueError, TypeError):
                pass

    if flag:
        is_abnormal = True

    return flag, is_abnormal


# =============================================================================
# Step 6: Generic patient info extraction
# =============================================================================

def extract_patient_info(full_text: str) -> dict:
    """Extract patient info generically from any report."""
    info = {}

    # Name patterns
    name_patterns = [
        r"(?:Patient\s*Detail[:\s]+)([\w\s\.]+?)(?:\n|Age|Registration)",
        r"(?:Mr\.|Mrs\.|Miss|Ms\.|Dr\.)\s+([\w\s]+?)(?:\n|Age|\.)",
        r"(?:Name[:\s]+)([\w\s\.]+?)(?:\n|Age|Gender)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip().rstrip(".")
            if 2 < len(name) < 50:
                info["name"] = name
                break

    # Age and gender
    age_match = re.search(
        r"(\d+)\s*(?:\(Y\)|Years?|Yrs?)\s*/?\s*([MFmf])\b",
        full_text
    )
    if age_match:
        info["age"]    = age_match.group(1)
        info["gender"] = "Female" if age_match.group(2).upper() == "F" else "Male"

    return info


# =============================================================================
# Step 7: Generic date extraction
# =============================================================================

def extract_dates(full_text: str) -> tuple[str, str]:
    """Extract collection and report dates generically."""
    DATE_RE = re.compile(
        r"\d{1,2}-[A-Za-z]{3}-\d{4}"
        r"|\d{1,2}/\d{1,2}/\d{4}"
        r"|\d{4}-\d{2}-\d{2}"
        r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
        re.IGNORECASE
    )

    collection_date = ""
    report_date     = ""

    lines = full_text.splitlines()
    for i, line in enumerate(lines):
        context = " ".join(lines[max(0, i-1):i+3])

        if re.search(r"collection", context, re.IGNORECASE):
            match = DATE_RE.search(context)
            if match and not collection_date:
                collection_date = match.group()

        if re.search(r"report(?:ing)?|verified", context, re.IGNORECASE):
            match = DATE_RE.search(context)
            if match and not report_date:
                report_date = match.group()

    # Fallback: any two dates found
    all_dates = DATE_RE.findall(full_text)
    if not collection_date and len(all_dates) > 0:
        collection_date = all_dates[0]
    if not report_date and len(all_dates) > 1:
        report_date = all_dates[1]

    return collection_date, report_date


# =============================================================================
# Step 8: Assemble final JSON
# =============================================================================

def build_json(
    full_text: str,
    elements:  list,
    source:    str,
) -> dict:
    """
    Assemble complete structured JSON from extracted text.
    Fully generic — works for any single-value chart report.
    """
    print("\nExtracting patient info...")
    patient_info = extract_patient_info(full_text)

    print("Extracting dates...")
    collection_date, report_date = extract_dates(full_text)

    print("Detecting tests...")
    detected_tests = detect_tests(full_text)

    sections = {}

    for test_info in detected_tests:
        print(f"\nExtracting: {test_info['name']}")

        value, unit = extract_value(full_text, test_info["keyword"])
        print(f"  Value: {value} {unit}")

        reference_range, _ = extract_reference_range(full_text)
        print(f"  Range: {reference_range}")

        flag, is_abnormal = detect_flag(full_text, value, reference_range)
        print(f"  Flag:  {flag} | Abnormal: {is_abnormal}")

        section_name = test_info["section"]
        if section_name not in sections:
            sections[section_name] = []

        sections[section_name].append({
            "test_name":       test_info["name"],
            "value":           value,
            "unit":            unit or test_info["unit"],
            "reference_range": reference_range,
            "flag":            flag,
            "is_abnormal":     is_abnormal,
        })

    return {
        "source_file":     source,
        "patient_info":    patient_info,
        "collection_date": collection_date,
        "report_date":     report_date,
        "sections":        sections,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/home/abbas/Documents/IMG_20260613_154648.pdf"

    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"\nParsing: {pdf_path}")
    print("=" * 60)

    # Extract text
    print("\nStep 1: Unstructured extraction...")
    elements, full_text = extract_text(pdf_path)

    print(f"\nRaw extracted text:")
    print("-" * 40)
    print(full_text)
    print("-" * 40)

    # Build JSON
    print("\nStep 2: Building structured JSON...")
    output = build_json(full_text, elements, Path(pdf_path).name)

    # Save
    output_path = Path(pdf_path).stem + "_parsed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("RESULT:")
    print("=" * 60)
    print(json.dumps(output, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()