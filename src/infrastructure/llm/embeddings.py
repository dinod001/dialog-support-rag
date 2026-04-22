"""
Embedding provider utilities.

Builds configured embedding instances using values loaded from settings.
"""

import importlib
from typing import Any
from langchain_openai import OpenAIEmbeddings

from infrastructure.config import settings

PROVIDER = settings.embedding.get("provider", "local")
EMBEDDING_MODEL = settings.embedding.get("model", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = settings.embedding.get(
    "local_model",
    "sentence-transformers/all-MiniLM-L6-v2",
)


class LocalSentenceTransformerEmbeddings:
    """Small adapter that exposes LangChain-style embedding methods."""

    def __init__(self, model_name: str) -> None:
        try:
            st_module = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(st_module, "SentenceTransformer")
        except Exception as exc:
            raise ImportError(
                "Local embeddings require 'sentence-transformers'. Install with: pip install sentence-transformers"
            ) from exc

        self._model = sentence_transformer_cls(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(vec) for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return list(vector)


def get_default_embeddings(
    show_progress: bool = False,
    **kwargs: Any
) -> Any:
    """
    Create an embeddings instance from configured defaults.

    Args:
        show_progress: Show progress bar during embedding.
        **kwargs: Additional arguments forwarded to provider-specific embedding class.

    Returns:
        A ready-to-use embeddings instance.
    """
    provider = str(PROVIDER).lower()

    if provider == "local":
        return LocalSentenceTransformerEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

    if provider == "groq":
        raise ValueError(
            "Groq is configured for chat LLM, not embeddings in this project. "
            "Use embedding.provider=openai or embedding.provider=local."
        )

    # Backward compatibility: some callers still pass batch_size, but
    # OpenAIEmbeddings expects chunk_size for request batching.
    if "batch_size" in kwargs and "chunk_size" not in kwargs:
        kwargs["chunk_size"] = kwargs.pop("batch_size")

    llm_kwargs: dict[str, Any] = dict(
        model=EMBEDDING_MODEL,
        show_progress_bar=show_progress,
        **kwargs,
    )

    llm_kwargs["openai_api_key"] = settings.OPENAI_API_KEY

    return OpenAIEmbeddings(**llm_kwargs)
