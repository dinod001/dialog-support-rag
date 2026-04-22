"""
Embedding provider utilities.

Builds a configured OpenAIEmbeddings instance using values loaded
from centralized project settings.
"""

from typing import Any
from langchain_openai import OpenAIEmbeddings

from infrastructure.config import settings

EMBEDDING_MODEL = settings.embedding.get("model", "text-embedding-3-small")
PROVIDER = settings.embedding.get("provider", "openai")


def get_default_embeddings(
    show_progress: bool = False,
    **kwargs: Any
) -> OpenAIEmbeddings:
    """
    Create an OpenAIEmbeddings instance from configured defaults.

    The model is resolved from settings.embedding with a safe fallback,
    and the OpenAI API key is injected from settings.

    Args:
        show_progress: Show progress bar during embedding.
        **kwargs: Additional arguments forwarded to OpenAIEmbeddings.

    Returns:
        A ready-to-use OpenAIEmbeddings instance.
    """
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
