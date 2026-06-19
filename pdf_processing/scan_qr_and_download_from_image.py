
# this one doesn't work properly

import cv2
from pyzbar.pyzbar import decode

def detect_qr(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not loaded")
        return

    h, w, _ = img.shape

    # ✅ Crop top-right area (where QR is in your image)
    qr_region = img[0:int(h*0.35), int(w*0.65):w]

    # ✅ Preprocess heavily
    gray = cv2.cvtColor(qr_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Sharpen
    kernel = [[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]
    kernel = cv2.UMat(kernel)
    gray = cv2.filter2D(gray, -1, kernel)

    # Threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Try decode
    qr_codes = decode(thresh)

    if qr_codes:
        for qr in qr_codes:
            data = qr.data.decode('utf-8')
            print("✅ QR Found:", data)
    else:
        print("❌ Still not detected")

# Run
detect_qr("/home/abbas/Downloads/Ami_fever_2026-20260612T142009Z-3-001/images/IMG_20260613_161037.jpg")