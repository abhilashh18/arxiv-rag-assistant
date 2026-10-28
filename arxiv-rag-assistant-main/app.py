import json
from pathlib import Path

import streamlit as st

from src.rag import answer_question

METADATA_PATH = Path("data/raw/metadata.json")


@st.cache_data
def load_paper_metadata():
    """
    Load paper titles/authors saved by arxiv_loader.py, so sources can
    show a readable title instead of a raw filename like
    '1811.08772v1.pdf'. Falls back gracefully if metadata.json doesn't
    exist (e.g. papers were added some other way).
    """
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def format_source_label(source, metadata):
    paper_info = metadata.get(source["paper"])
    label = paper_info["title"] if paper_info else source["paper"]

    page_start = source["page"]
    page_end = source.get("page_end", page_start)
    page_label = f"p.{page_start}" if page_end == page_start else f"pp.{page_start}-{page_end}"

    return f"{label} — {page_label} (score: {source['score']:.3f})"


st.title("📚 arXiv RAG Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

metadata = load_paper_metadata()

question = st.text_input("Ask a question about the research papers:")

if st.button("Ask") and question:

    with st.spinner("Searching papers and generating answer..."):
        try:
            answer, sources = answer_question(question)
            error = None
        except FileNotFoundError as exc:
            answer, sources = None, []
            error = (
                "The paper index isn't set up yet. Run "
                "`python -m src.ingestion.arxiv_loader`, then "
                "`python -m src.chunking.chunking`, then "
                f"`python -m src.retrieval.build_index` first.\n\nDetails: {exc}"
            )
        except Exception as exc:
            answer, sources = None, []
            error = f"Something went wrong answering that question: {exc}"

    if error:
        st.error(error)
    else:
        st.session_state.history.append((question, answer, sources))

# Show most recent answer first, then earlier ones below it.
for past_question, past_answer, past_sources in reversed(st.session_state.history):
    st.subheader(f"Q: {past_question}")
    st.write(past_answer)

    if past_sources:
        st.caption("Sources")
        for source in past_sources:
            st.write(f"- {format_source_label(source, metadata)}")

    st.divider()