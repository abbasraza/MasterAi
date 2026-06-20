#!/usr/bin/env python3
"""
Detect images and charts in lab report PDFs.
Usage: python detect_charts.py <pdf_path_or_folder>
"""

from pathlib import Path
import sys


def detect_charts(pdf_path: Path) -> dict:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result    = converter.convert(str(pdf_path))
    doc       = result.document
    markdown  = doc.export_to_markdown()
    print(" markdown is ", markdown)

    n_pages  = len(doc.pages)
    n_tables = len(doc.tables)

    # ── Safe image extraction (handles different Docling versions) ─────────
    n_images    = 0
    image_types = []

    # Try different attribute names across Docling versions
    image_sources = [
        ("images",        lambda d: d.images),
        ("pictures",      lambda d: d.pictures),
        ("figures",       lambda d: d.figures),
    ]

    images_list = []
    for attr_name, getter in image_sources:
        try:
            imgs = getter(doc)
            if imgs:
                images_list = list(imgs)
                break
        except AttributeError:
            continue

    n_images = len(images_list)

    for i, image in enumerate(images_list):
        try:
            label = (
                getattr(image, "label", None) or
                getattr(image, "type",  None) or
                getattr(image, "kind",  None)
            )
            size = getattr(image, "size", None)
            image_types.append({
                "index": i,
                "label": str(label) if label else "unknown",
                "size":  str(size)  if size  else "unknown",
            })
        except Exception:
            image_types.append({"index": i, "label": "unknown"})

    # ── Count image placeholders in markdown (most reliable) ──────────────
    import re
    image_placeholders = re.findall(r"<!-- image -->", markdown)

    # Use placeholder count as fallback if no images found via API
    if n_images == 0 and image_placeholders:
        n_images = len(image_placeholders)

    # ── Rest unchanged ─────────────────────────────────────────────────────
    import re
    CHART_KEYWORDS    = [
        "esr", "erythrocyte sedimentation",
        "mm/1st", "westergren",
        "hba1c", "glycated hemoglobin",
        "trend", "graph", "chart",
    ]
    text_lower        = markdown.lower()
    matched_keywords  = [kw for kw in CHART_KEYWORDS if kw in text_lower]
    is_chart_report   = n_images > 0 and n_tables == 0
    is_mixed_report   = n_images > 0 and n_tables > 0

    return {
        "file":               pdf_path.name,
        "pages":              n_pages,
        "tables":             n_tables,
        "images":             n_images,
        "image_placeholders": len(image_placeholders),
        "image_types":        image_types,
        "is_chart_report":    is_chart_report,
        "is_mixed_report":    is_mixed_report,
        "chart_keywords":     matched_keywords,
        "verdict":            _verdict(n_images, n_tables, matched_keywords),
        "markdown_preview":   markdown[:500],
    }
    
def _verdict(n_images: int, n_tables: int, keywords: list) -> str:
    """Human readable verdict."""
    if n_tables > 0 and n_images == 0:
        return "TABLE ONLY — standard parser, no LLM needed"
    if n_tables > 0 and n_images > 0:
        return "MIXED — table parser + check images manually"
    if n_images > 0 and keywords:
        return f"CHART REPORT — LLM needed (keywords: {', '.join(keywords)})"
    if n_images > 0:
        return "IMAGE ONLY — likely chart, LLM needed"
    if n_tables == 0 and n_images == 0:
        return "FREE TEXT — LLM needed"
    return "UNKNOWN"


def print_result(result: dict):
    print(f"\n{'=' * 60}")
    print(f"  {result['file']}")
    print(f"{'=' * 60}")
    print(f"  Pages             : {result['pages']}")
    print(f"  Tables            : {result['tables']}")
    print(f"  Images            : {result['images']}")
    print(f"  Image placeholders: {result['image_placeholders']}")
    print(f"  Chart keywords    : {result['chart_keywords'] or 'none'}")
    print(f"  Is chart report   : {result['is_chart_report']}")
    print(f"  Is mixed report   : {result['is_mixed_report']}")
    print(f"\n  VERDICT: {result['verdict']}")

    if result["image_types"]:
        print(f"\n  Images found:")
        for img in result["image_types"]:
            print(
                f"    [{img['index']}] "
                f"label={img['label']} "
                f"size={img['size']}"
            )

    print(f"\n  Markdown preview:")
    print(f"  {'-' * 40}")
    for line in result["markdown_preview"].splitlines()[:10]:
        print(f"  {line}")
    print(f"  {'-' * 40}")


def run(source: str):
    path = Path(source)

    if path.is_file():
        pdf_files = [path]
    elif path.is_dir():
        pdf_files = sorted(path.rglob("*.pdf"))
    else:
        print(f"Error: path not found: {source}")
        sys.exit(1)

    if not pdf_files:
        print(f"No PDFs found in: {source}")
        sys.exit(1)

    print(f"\nScanning {len(pdf_files)} PDF(s)...")

    # ── Summary counters ───────────────────────────────────────────────────
    summary = {
        "table_only":   [],
        "chart_report": [],
        "mixed":        [],
        "free_text":    [],
        "unknown":      [],
    }

    for pdf_file in pdf_files:
        try:
            result = detect_charts(pdf_file)
            print_result(result)

            # Categorize
            v = result["verdict"]
            if "TABLE ONLY"    in v: summary["table_only"].append(pdf_file.name)
            elif "CHART REPORT" in v: summary["chart_report"].append(pdf_file.name)
            elif "MIXED"        in v: summary["mixed"].append(pdf_file.name)
            elif "FREE TEXT"    in v: summary["free_text"].append(pdf_file.name)
            else:                     summary["unknown"].append(pdf_file.name)

        except Exception as e:
            print(f"\nFailed: {pdf_file.name}: {e}")

    # ── Print batch summary ────────────────────────────────────────────────
    if len(pdf_files) > 1:
        print(f"\n{'=' * 60}")
        print(f"  BATCH SUMMARY ({len(pdf_files)} PDFs)")
        print(f"{'=' * 60}")
        print(f"  Table only   (free)  : {len(summary['table_only'])}")
        print(f"  Chart report (LLM)   : {len(summary['chart_report'])}")
        print(f"  Mixed                : {len(summary['mixed'])}")
        print(f"  Free text   (LLM)    : {len(summary['free_text'])}")
        print(f"  Unknown     (LLM)    : {len(summary['unknown'])}")

        if summary["chart_report"]:
            print(f"\n  Chart reports (need LLM):")
            for f in summary["chart_report"]:
                print(f"    - {f}")

        if summary["free_text"]:
            print(f"\n  Free text reports (need LLM):")
            for f in summary["free_text"]:
                print(f"    - {f}")

        total_llm = (
            len(summary["chart_report"]) +
            len(summary["free_text"])    +
            len(summary["unknown"])
        )
        print(f"\n  Will use LLM : {total_llm}/{len(pdf_files)} PDFs")
        print(f"  Free (regex) : {len(summary['table_only'])}/{len(pdf_files)} PDFs")
        print(f"{'=' * 60}")

def debug_doc_attributes(pdf_path: Path):
    """Run this once to see what your Docling version exposes."""
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(str(pdf_path)).document

    print(f"\nDocling version attributes for: {pdf_path.name}")
    print(f"  All attributes: {[a for a in dir(doc) if not a.startswith('_')]}")

    # Check each image-related attribute
    for attr in ["images", "pictures", "figures", "elements"]:
        try:
            val = getattr(doc, attr)
            print(f"  {attr}: {len(list(val))} items")
        except AttributeError:
            print(f"  {attr}: not available")


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1])
    debug_doc_attributes(path)   # run this first
    run(sys.argv[1])
    
