"""
Quick manual check: does the column-aware extraction in pdf_loader.py
actually change anything for your real downloaded papers, compared to
PyMuPDF's default get_text()?

Usage:
    python verify_extraction.py data/raw/<some_paper>.pdf
    python verify_extraction.py data/raw/<some_paper>.pdf --page 2
"""

import argparse
import pymupdf

from src.ingestion.pdf_loader import extract_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to a PDF in data/raw/")
    parser.add_argument(
        "--page", type=int, default=1,
        help="1-indexed page number to compare (default: 1)"
    )
    args = parser.parse_args()

    page_index = args.page - 1

    # OLD behavior: PyMuPDF's default full-width reading order
    doc = pymupdf.open(args.pdf_path)
    old_text = doc[page_index].get_text()
    doc.close()

    # NEW behavior: our column-aware extraction
    pages = extract_pages(args.pdf_path)
    new_text = pages[page_index]["text"]

    print(f"Comparing page {args.page} of {args.pdf_path}\n")

    if old_text.strip() == new_text.strip():
        print("SAME — the fix made no difference on this page.")
    else:
        print("DIFFERENT — the fix changed extraction on this page.\n")
        print("=" * 60)
        print("OLD (default get_text):")
        print("=" * 60)
        print(old_text)
        print("\n" + "=" * 60)
        print("NEW (column-aware extract_pages):")
        print("=" * 60)
        print(new_text)


if __name__ == "__main__":
    main()