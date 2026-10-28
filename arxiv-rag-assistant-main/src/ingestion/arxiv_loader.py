import argparse
import json
import time
from pathlib import Path

import arxiv
import requests

# Polite delay between PDF downloads so we don't hammer arXiv's servers.
DOWNLOAD_DELAY_SECONDS = 3
# Retry settings for flaky network calls.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


def _download_with_retries(url, max_retries=MAX_RETRIES):
    """GET a URL, retrying with exponential backoff on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response
        except (requests.RequestException,) as exc:
            if attempt == max_retries:
                raise
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"  Download failed ({exc}); retrying in {wait}s "
                  f"(attempt {attempt}/{max_retries})...")
            time.sleep(wait)


def download_papers(query, max_results=10):
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    downloaded_count = 0

    for result in client.results(search):

        paper_id = result.get_short_id().replace("/", "_")
        filename = output_dir / f"{paper_id}.pdf"

        # Always record/refresh metadata, even for already-downloaded papers,
        # so older downloads (from before this field existed) get backfilled.
        metadata[filename.name] = {
            "paper_id": paper_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.isoformat() if result.published else None,
            "pdf_url": result.pdf_url,
            "entry_id": result.entry_id,
        }

        if filename.exists():
            print(f"Already exists: {filename.name}")
            continue

        print(f"Downloading: {result.title}")

        response = _download_with_retries(result.pdf_url)

        with open(filename, "wb") as f:
            f.write(response.content)

        downloaded_count += 1

        # Save metadata incrementally so a crash mid-run doesn't lose
        # everything gathered so far.
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Be polite to arXiv's servers between downloads.
        time.sleep(DOWNLOAD_DELAY_SECONDS)

    # Final metadata save covers the case where every paper was already
    # downloaded (loop above never re-saves for skip-only runs).
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Done. Downloaded {downloaded_count} new paper(s); "
          f"metadata for {len(metadata)} paper(s) saved to {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download papers from arXiv for the RAG pipeline."
    )
    parser.add_argument(
        "--query",
        default="retrieval augmented generation",
        help="arXiv search query (default: 'retrieval augmented generation')",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of papers to download (default: 20)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_papers(query=args.query, max_results=args.max_results)