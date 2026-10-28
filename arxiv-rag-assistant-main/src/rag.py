from src.retrieval.retriever import retrieve_chunks
from src.generation.llm import generate_answer, NO_CONTEXT_ANSWER

# Chunks scoring below this cosine similarity are treated as "not
# actually relevant" and excluded from context -- without this, the
# assistant will confidently answer using whatever top-5 chunks it got
# back even if none of them are actually related to the question.
MIN_RELEVANCE_SCORE = 0.35


def _format_page_range(chunk):
    start = chunk["page_number"]
    end = chunk.get("page_number_end", start)
    return f"p.{start}" if end == start else f"pp.{start}-{end}"


def answer_question(question):

    results = retrieve_chunks(question, top_k=5)

    relevant_results = [r for r in results if r["score"] >= MIN_RELEVANCE_SCORE]

    if not relevant_results:
        return NO_CONTEXT_ANSWER, []

    # Label each chunk with its source so the model can (at least in
    # principle) attribute claims to a specific paper, instead of seeing
    # one undifferentiated blob of text.
    context = "\n\n".join(
        f"[Source: {result['chunk']['paper']}, {_format_page_range(result['chunk'])}]\n"
        f"{result['chunk']['text']}"
        for result in relevant_results
    )

    answer = generate_answer(
        context=context,
        question=question
    )

    sources = []

    for result in relevant_results:
        chunk = result["chunk"]

        sources.append({
            "paper": chunk["paper"],
            "page": chunk["page_number"],
            "page_end": chunk.get("page_number_end", chunk["page_number"]),
            "score": result["score"]
        })

    return answer, sources


if __name__ == "__main__":

    question = "What is retrieval augmented generation?"

    answer, sources = answer_question(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for source in sources:
        print(
            f"- {source['paper']} | "
            f"Page {source['page']} | "
            f"Score {source['score']:.3f}"
        )