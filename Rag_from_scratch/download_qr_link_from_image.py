from qreader import QReader
import cv2
import requests

# Initialize reader
qreader = QReader()

# Load image
image = cv2.imread("/home/abbas/Downloads/Ami_fever_2026-20260612T142009Z-3-001/images/IMG_20260613_161037.jpg")

if image is None:
    print("❌ Error: Image not loaded")
    exit()

# Detect + decode
decoded_texts = qreader.detect_and_decode(image=image)

print("QR Output:", decoded_texts)

if not decoded_texts:
    print("❌ No QR detected")
    exit()

tiny_url = decoded_texts[0]
print("🔗 TinyURL:", tiny_url)

try:
    # Step 1: Extract redirect URL
    head_resp = requests.head(
        tiny_url,
        allow_redirects=False,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10
    )

    redirect_url = (
        head_resp.headers.get("Location") or
        head_resp.headers.get("x-tinyurl-target")
    )

    if not redirect_url:
        print("❌ No redirect URL found")
        exit()

    print("➡️ Redirect URL:", redirect_url)

    # --- Download with timeout + protocol fallback ---
    def robust_download(url):
        try:
            print(f"⬇️ Trying: {url}")
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            return resp
        except requests.exceptions.Timeout:
            print("⚠️ Timeout occurred")

            # Switch protocol
            if url.startswith("http://"):
                secure_url = url.replace("http://", "https://")
                print(f"🔁 Retrying with HTTPS: {secure_url}")
            elif url.startswith("https://"):
                secure_url = url.replace("https://", "http://")
                print(f"🔁 Retrying with HTTP: {secure_url}")
            else:
                raise

            # Retry once with switched protocol
            return requests.get(
                secure_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )

    # Step 2: Perform download
    response = robust_download(redirect_url)

    if response.status_code != 200:
        print(f"❌ Download failed (status {response.status_code})")
        exit()

    # Step 3: Save as PDF
    with open("report.pdf", "wb") as f:
        f.write(response.content)

    print("✅ Saved as report.pdf")

except Exception as e:
    print("❌ Error:", e)