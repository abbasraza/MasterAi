import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

input_dir = "/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images/failed_downloads"
output_dir = "/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images/failed_downloads/pdfs"

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
        # prompt do you want to continue with this file or skip it? (y/n)
        cont = input("Continue with this file? (y/n): ")
        if cont.lower() != 'y':
            os.remove(pdf_path)
            print(f"Skipped and removed: {pdf_name}")

print("Done ✅")
