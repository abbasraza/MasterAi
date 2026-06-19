from pathlib import Path
from docling.document_converter import DocumentConverter

pdf_path = Path("/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images/downloaded_pdfs/manually_downloaded/20260614103618570_6c38c284-a683-40b6-9c34-c9bfec06d15b.pdf")

converter = DocumentConverter()
result    = converter.convert(str(pdf_path))
doc       = result.document

# ── Option 1: Full document as dict ───────────────────────────────────────
import json
doc_dict = doc.export_to_dict()
with open("docling_full.json", "w") as f:
    json.dump(doc_dict, f, indent=2)
print("Saved docling_full.json")

# ── Option 2: Markdown (tables rendered as markdown) ──────────────────────
with open("docling_output.md", "w") as f:
    f.write(doc.export_to_markdown())
print("Saved docling_output.md")

# ── Option 3: HTML (tables as proper HTML <table>) ────────────────────────
with open("docling_output.html", "w") as f:
    f.write(doc.export_to_html())
print("Saved docling_output.html")

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