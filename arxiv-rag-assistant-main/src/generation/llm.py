import ollama

MODEL_NAME = "llama3.2:3b"

NO_CONTEXT_ANSWER = (
    "I don't have any relevant information in the paper corpus to answer that."
)


def generate_answer(context, question, temperature=0.2, max_tokens=512):
    if not context or not context.strip():
        # No retrieved chunks -- answering anyway would just be the model
        # making something up from its own general knowledge, which
        # defeats the point of grounding answers in your paper corpus.
        return NO_CONTEXT_ANSWER

    prompt = f"""
Answer the question using ONLY the context below. If the context does
not contain enough information to answer, respond with exactly:
"{NO_CONTEXT_ANSWER}"
Do not use any outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
    except ConnectionError as exc:
        return (
            "Couldn't reach the local Ollama server. Make sure Ollama is "
            f"running (`ollama serve`). Details: {exc}"
        )
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            return (
                f"Model '{MODEL_NAME}' isn't pulled yet. Run "
                f"`ollama pull {MODEL_NAME}` and try again."
            )
        return f"Ollama returned an error: {exc}"
    except Exception as exc:
        return f"Unexpected error generating an answer: {exc}"

    return response["message"]["content"]


if __name__ == "__main__":
    context = "RAG combines information retrieval with language generation."

    answer = generate_answer(
        context,
        "What is RAG?"
    )

    print(answer)


if __name__ == "__main__":
    context = "RAG combines information retrieval with language generation."

    answer = generate_answer(
        context,
        "What is RAG?"
    )

    print(answer)