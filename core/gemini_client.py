import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

_client = None


def _get_client():
    global _client
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file."
        )

    if _client is None:
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30_000),  # milliseconds
        )
    return _client


def _extract_text(response):
    text = getattr(response, "text", None)
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return part_text

    raise ValueError("Gemini response did not include any text output.")


def _extract_embeddings(response):
    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        raise ValueError("Gemini returned no embeddings.")

    values = []
    for embedding in embeddings:
        embedding_values = getattr(embedding, "values", None)
        if embedding_values is None:
            raise ValueError("Embedding response did not contain vector values.")
        values.append(list(embedding_values))
    return values


def get_embeddings(text_chunks):
    response = _get_client().models.embed_content(
        model="gemini-embedding-001",
        contents=list(text_chunks),
    )
    return _extract_embeddings(response)


def get_query_embedding(query):
    response = _get_client().models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    embeddings = _extract_embeddings(response)
    return embeddings[0]


def generate_answer(query, context_chunks):
    context_string = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
    Answer the following query using strictly the context provided below.
    If the context does not contain the answer, clearly state that you do not have enough information.

    Context:
    {context_string}

    Query: {query}
    """

    response = _get_client().models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return _extract_text(response)