import pdfplumber
import sys

sys.stdout.reconfigure(encoding="utf-8")

pdf_path = "C:/Users/Admin/Downloads/main.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print("pages", len(pdf.pages))
    for i, page in enumerate(pdf.pages[:8], start=1):
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        print(f"---PAGE {i}---")
        print("\n".join(lines[:30]))
