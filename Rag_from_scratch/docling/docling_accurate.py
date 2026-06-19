from pathlib import Path
import json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ---------------------------------------------------------------------------
# Build converter with all table options enabled
# ---------------------------------------------------------------------------

pipeline_options = PdfPipelineOptions()

# ── Table structure ────────────────────────────────────────────────────────
pipeline_options.do_table_structure = True

# ── OCR ───────────────────────────────────────────────────────────────────
pipeline_options.do_ocr = True

# ── Generate images (required for ACCURATE mode) ──────────────────────────
pipeline_options.generate_page_images  = True
pipeline_options.generate_table_images = True

# ── TableFormer ACCURATE mode + cell matching ──────────────────────────────
# Try the current Docling v2 API
try:
    from docling.datamodel.pipeline_options import TableStructureOptions, TableFormerMode

    pipeline_options.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE,
        do_cell_matching=True,
    )
    print(" TableFormer ACCURATE + cell matching enabled")

except ImportError:
    # Older Docling API
    try:
        from docling.datamodel.pipeline_options import TableFormerMode
        pipeline_options.table_structure_options = {
            "mode": TableFormerMode.ACCURATE,
            "do_cell_matching": True,
        }
        print(" TableFormer ACCURATE (legacy API)")

    except Exception as e:
        print(f"  Could not set ACCURATE mode: {e}")
        print("   Running with default table structure")


converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------

pdf_path = Path("/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images/downloaded_pdfs/manually_downloaded/20260614103618570_6c38c284-a683-40b6-9c34-c9bfec06d15b.pdf")

print(f"\nConverting: {pdf_path.name}")
result = converter.convert(str(pdf_path))
doc    = result.document


# ---------------------------------------------------------------------------
# Export all formats
# ---------------------------------------------------------------------------

# ── 1. Full document dict ──────────────────────────────────────────────────
doc_dict = doc.export_to_dict()
with open("docling_full.json", "w", encoding="utf-8") as f:
    json.dump(doc_dict, f, indent=2, ensure_ascii=False)
print(" Saved docling_full.json")

# ── 2. Markdown ────────────────────────────────────────────────────────────
with open("docling_output.md", "w", encoding="utf-8") as f:
    f.write(doc.export_to_markdown())
print(" Saved docling_output.md")

# ── 3. HTML ────────────────────────────────────────────────────────────────
with open("docling_output.html", "w", encoding="utf-8") as f:
    f.write(doc.export_to_html())
print(" Saved docling_output.html")


# ---------------------------------------------------------------------------
# Quick debug — show raw table columns and first few rows
# ---------------------------------------------------------------------------

print(f"\n── Tables found: {len(doc.tables)} ──")
for i, table in enumerate(doc.tables):
    print(f"\n  Table {i}:")
    try:
        df = table.export_to_dataframe()
    except TypeError:
        try:
            df = table.export_to_dataframe(doc)
        except Exception as e:
            print(f"    Export failed: {e}")
            continue
    except Exception as e:
        print(f"    Export failed: {e}")
        continue

    print(f"    Columns : {list(df.columns)}")
    print(f"    Shape   : {df.shape}")
    print(df.to_string(index=False))