import pdfplumber

def debug_spatial_extraction(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=False,
            )

            rows = {}
            for word in words:
                y_bucket = round(word["top"] / 5) * 5
                rows.setdefault(y_bucket, []).append(
                    (word["x0"], word["text"])
                )

            print(f"\n{'='*60}")
            print(f"PAGE {page_num}")
            print(f"{'='*60}")
            for y in sorted(rows.keys()):
                line_words = sorted(rows[y], key=lambda w: w[0])
                line_text  = "  ".join(w[1] for w in line_words)
                print(line_text)

# Run it
debug_spatial_extraction("/home/abbas/Downloads/Ami_fever_2026-20260612T142009Z-3-001/Ami_fever_2026/ami_gram_stain.pdf")