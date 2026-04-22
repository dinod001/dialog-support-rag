"""
LLM provider factory utilities.

Centralizes creation of ChatOpenAI instances using project settings
for API credentials and runtime model parameters.
"""

from typing import Optional, Any
from langchain_openai import ChatOpenAI

from infrastructure.config import settings

OPENAI_API_KEY = settings.OPENAI_API_KEY


def _build_llm(
    model: str,
    temperature: float = 0,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Internal factory — builds a ChatOpenAI for any provider."""
    llm_kwargs: dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens,
        **kwargs,
    )

    llm_kwargs["openai_api_key"] = OPENAI_API_KEY

    return ChatOpenAI(**llm_kwargs)
