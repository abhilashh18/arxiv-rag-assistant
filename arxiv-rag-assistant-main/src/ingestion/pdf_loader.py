import pymupdf


def _sort_blocks_by_reading_order(blocks, page_width):
    """
    Reorder text blocks into correct reading order for single- or
    two-column academic layouts.

    PyMuPDF's default page.get_text() reads left-to-right across the
    full page width, which interleaves columns mid-sentence on
    two-column papers (the vast majority of arXiv PDFs). Instead, we
    split blocks into a left and right half based on the page's
    horizontal midpoint, sort each half top-to-bottom, and concatenate
    left-column-then-right-column. Single-column pages still work
    correctly, since everything falls into one "column" and gets
    sorted top-to-bottom as normal.
    """
    midpoint = page_width / 2

    left_column = []
    right_column = []

    for block in blocks:
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if not text.strip():
            continue

        block_center_x = (x0 + x1) / 2
        if block_center_x < midpoint:
            left_column.append((y0, text))
        else:
            right_column.append((y0, text))

    left_column.sort(key=lambda item: item[0])
    right_column.sort(key=lambda item: item[0])

    ordered_text = [text for _, text in left_column] + \
        [text for _, text in right_column]

    return "\n".join(ordered_text)


def extract_pages(pdf_path):
    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as exc:
        print(f"  Skipping unreadable PDF {pdf_path}: {exc}")
        return []

    pages = []

    try:
        for page_number, page in enumerate(pdf):
            blocks = page.get_text("blocks")
            page_text = _sort_blocks_by_reading_order(blocks, page.rect.width)

            if not page_text.strip():
                print(f"  Warning: {pdf_path} page {page_number + 1} "
                      f"has no extractable text (likely a scanned image).")

            pages.append({
                "page_number": page_number + 1,
                "text": page_text
            })
    finally:
        pdf.close()

    return pages