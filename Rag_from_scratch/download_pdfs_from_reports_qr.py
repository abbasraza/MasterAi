from qreader import QReader
import cv2
import requests
import os
import shutil

# -------- CONFIG --------
INPUT_DIR = "/home/abbas/Downloads/Images-20260614T073049Z-3-001/Images"

GOOD_DIR = os.path.join(INPUT_DIR, "good_images")
NO_QR_DIR = os.path.join(INPUT_DIR, "no_qr_images")
FAILED_DIR = os.path.join(INPUT_DIR, "failed_downloads")
PDF_DIR = os.path.join(INPUT_DIR, "downloaded_pdfs")

FAILED_URLS_FILE = os.path.join(INPUT_DIR, "failed_urls.txt")

# Create directories
os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(NO_QR_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# Initialize QR reader
qreader = QReader()

from playwright.sync_api import sync_playwright

def download_via_playwright(url, output_path):
    print(f"🌐 Playwright fallback for: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(url, timeout=60000)

            # wait a bit for JS to fully render PDF
            page.wait_for_timeout(5000)

            # Save as PDF
            page.pdf(path=output_path)

            browser.close()

        print("✅ Saved via Playwright:", output_path)
        return True

    except Exception as e:
        print("❌ Playwright failed:", e)
        return False

# -------- Helper: Robust download --------
def robust_download(url):
    try:
        print(f"⬇️ Trying: {url}")
        return requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )
    except requests.exceptions.Timeout:
        print("⚠️ Timeout occurred")

        # Switch protocol
        if url.startswith("http://"):
            url = url.replace("http://", "https://")
        elif url.startswith("https://"):
            url = url.replace("https://", "http://")

        print(f"🔁 Retrying: {url}")
        return requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )


# -------- Helper: Save failed URL --------
def log_failed_url(image_name, url, tiny_url=None):
    with open(FAILED_URLS_FILE, "a") as f:
        if tiny_url:
            f.write(f"{image_name} -> {url} (TinyURL: {tiny_url})\n")
        else:
            f.write(f"{image_name} -> {url}\n")


# -------- Process images --------
for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    print(f"\n📷 Processing: {filename}")
    tiny_url = None  # Initialize tiny_url for logging in case of early failure
    try:
        image = cv2.imread(img_path)

        if image is None:
            print("❌ Image load failed")
            shutil.move(img_path, os.path.join(NO_QR_DIR, filename))
            continue

        decoded_texts = qreader.detect_and_decode(image=image)

        # -------- Case 1: No QR --------
        if not decoded_texts:
            print("❌ No QR found")
            shutil.move(img_path, os.path.join(NO_QR_DIR, filename))
            continue

        tiny_url = decoded_texts[0]
        print("🔗 TinyURL:", tiny_url)

        # -------- Get redirect URL --------
        # -------- Get redirect URL --------
        redirect_url = None

        try:
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

        except Exception as e:
            print("⚠️ HEAD request failed:", e)

        # -------- Fallback if no redirect --------
        if not redirect_url:
            print("⚠️ No redirect from HEAD, trying GET fallback...")

            try:
                get_resp = requests.get(
                    tiny_url,
                    allow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15
                )

                redirect_url = get_resp.url
                print("✅ Resolved via GET:", redirect_url)

            except Exception as e:
                print("❌ Failed to resolve URL:", e)
                log_failed_url(filename, tiny_url, tiny_url)
                shutil.move(img_path, os.path.join(FAILED_DIR, filename))
                continue

        print("➡️ Final URL:", redirect_url)

        # -------- Download --------
        response = robust_download(redirect_url)

        if response.status_code != 200:
            print(f"❌ Download failed ({response.status_code})")
            log_failed_url(filename, redirect_url, tiny_url)
            shutil.move(img_path, os.path.join(FAILED_DIR, filename))
            continue

        # -------- Save PDF --------
        pdf_name = os.path.splitext(filename)[0] + ".pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_name)

        # -------- Optional PDF check --------
        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower():
            print("⚠️ Not a PDF (likely HTML)")
       
            # success = download_via_playwright(redirect_url, pdf_path)
        
            # if success:
            #     shutil.move(img_path, os.path.join(GOOD_DIR, filename))
            # else:
            log_failed_url(filename, redirect_url, tiny_url)
            shutil.move(img_path, os.path.join(FAILED_DIR, filename))

            continue


        with open(pdf_path, "wb") as f:
            f.write(response.content)

        print(f"✅ Saved PDF: {pdf_name}")

        # Move to good_images
        shutil.move(img_path, os.path.join(GOOD_DIR, filename))

    except Exception as e:
        print("❌ Error:", e)
        log_failed_url(filename, "UNKNOWN_ERROR", tiny_url)
        shutil.move(img_path, os.path.join(FAILED_DIR, filename))


print("\n✅ Processing complete!")
print(f"📄 Failed URLs logged in: {FAILED_URLS_FILE}")
