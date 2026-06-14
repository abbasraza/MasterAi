import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

input_dir = "/home/abbas/Downloads/Ami_fever_2026-20260612T142009Z-3-001/images"
output_dir = "/home/abbas/Downloads/Ami_fever_2026-20260612T142009Z-3-001/pdfs"




def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_dir, file)
        pdf_name = os.path.splitext(file)[0] + ".pdf"
        pdf_path = os.path.join(output_dir, pdf_name)

        img = Image.open(img_path)

        pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf')

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        print(f"OCR Converted: {file} → {pdf_name}")
        text = extract_text(pdf_path)  # Optional: Extract text to verify OCR output
        print(f"Extracted Text: {text}...")  # Print the first 100 characters of the extracted text

print("Done ✅")
text = extract_text("/home/abbas/Downloads/Ami_fever_2026-20260612T142009Z-3-001/pdfs/high_quality_report.pdf")
print(f"Another text: {text}...")  # Print the first 100 characters of the extracted text