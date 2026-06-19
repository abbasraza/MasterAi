import json
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,      #  correct class in v2.x
)
from docling.datamodel.base_models import InputFormat


def build_converter() -> DocumentConverter:

    # ── Correct way to set TableFormer mode in Docling v2.x ───────────────
    table_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE,   #  ML-based table parsing
        do_cell_matching=True,
    )

    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        table_structure_options=table_options,   #  pass object, not dict
        do_ocr=True,
        generate_page_images=True,
        generate_table_images=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    return converter


def extract_lab_data(pdf_path: Path) -> tuple[dict, str]:
    converter = build_converter()
    result    = converter.convert(str(pdf_path))
    doc       = result.document

    output = {
        "patient_info": {},
        "sections":     [],
    }

    markdown = doc.export_to_markdown()

    # ── Patient info from key-value pairs ──────────────────────────────────
    for kv in doc.key_value_items:
        try:
            key   = kv.key.text.strip().lower()   if kv.key   else ""
            value = kv.value.text.strip()         if kv.value else ""
            if not value:
                continue
            if "name"                    in key:
                output["patient_info"]["name"]        = value
            elif "age"                   in key:
                output["patient_info"]["age"]         = value
            elif "gender" in key or "sex" in key:
                output["patient_info"]["gender"]      = value
            elif "date" in key or "collected" in key or "reported" in key:
                output["patient_info"]["report_date"] = value
        except Exception:
            continue

    # ── Tables ─────────────────────────────────────────────────────────────
    for i, table in enumerate(doc.tables):

        # Section name from parent heading
        section_name = "General Lab Tests"
        try:
            parent = table.parent
            if parent and hasattr(parent, "text") and parent.text:
                section_name = parent.text.strip()
        except Exception:
            pass

        # Export table
        try:
            df = table.export_to_dataframe()
        except TypeError:
            try:
                df = table.export_to_dataframe(doc)
            except Exception:
                continue
        except Exception:
            continue

        if df is None or df.empty:
            continue

        tests = []
        for record in df.to_dict(orient="records"):
            test = {
                "test_name":       None,
                "value":           None,
                "unit":            None,
                "reference_range": None,
                "flag":            None,
                "is_abnormal":     False,
            }

            for col, val in record.items():
                col_lower = str(col).lower().strip()
                val_str   = str(val).strip() if val is not None else ""

                if not val_str or val_str.lower() in {"none", "nan", "-", ""}:
                    continue

                if any(k in col_lower for k in ["test", "parameter", "investigation", "analyte", "name"]):
                    test["test_name"] = val_str
                elif any(k in col_lower for k in ["result", "value", "finding"]):
                    test["value"] = val_str
                elif "unit" in col_lower:
                    test["unit"] = val_str
                elif any(k in col_lower for k in ["ref", "range", "normal", "interval"]):
                    test["reference_range"] = val_str
                elif "flag" in col_lower:
                    test["flag"] = val_str

            if test["test_name"]:
                tests.append(test)

        if tests:
            output["sections"].append({
                "section_name": section_name,
                "tests":        tests,
            })

    return output, markdown


if __name__ == "__main__":
    pdf_path = Path("/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images/downloaded_pdfs/manually_downloaded/20260614103618570_6c38c284-a683-40b6-9c34-c9bfec06d15b.pdf")

    print("Converting with Docling ACCURATE mode...")
    result, markdown = extract_lab_data(pdf_path)

    with open("structured_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with open("structured_output.md", "w", encoding="utf-8") as f:
        f.write(markdown)

    print("✅ Done!")
    print(f"   Patient : {result['patient_info']}")
    print(f"   Sections: {[s['section_name'] for s in result['sections']]}")