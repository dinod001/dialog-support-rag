"""
LLM provider factory utilities.

Centralizes creation of chat model instances using project settings
for API credentials and runtime model parameters.
"""

from typing import Optional, Any
import importlib
from langchain_openai import ChatOpenAI

from infrastructure.config import settings

OPENAI_API_KEY = settings.OPENAI_API_KEY
GROQ_API_KEY = settings.GROQ_API_KEY


def _resolve_model(provider: str, model: Optional[str]) -> str:
    if model:
        return model

    provider_models = settings.models.get(provider, {}).get("chat", {})
    tier = settings.llm.get("model_tier", "general")
    if tier in provider_models:
        return provider_models[tier]
    if "general" in provider_models:
        return provider_models["general"]

    # Final fallback keeps legacy behavior.
    return settings.models.get("openai", {}).get("chat", {}).get("general", "gpt-4o-mini")


def _build_llm(
    provider: str,
    model: str,
    temperature: float = 0,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Internal factory — builds a provider-specific chat model."""
    llm_kwargs: dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens,
        **kwargs,
    )

    normalized = provider.lower()
    if normalized == "groq":
        try:
            chat_groq_module = importlib.import_module("langchain_groq")
            chat_groq_cls = getattr(chat_groq_module, "ChatGroq")
        except Exception as exc:
            raise ImportError(
                "Groq support requires 'langchain-groq'. Install it with: pip install langchain-groq"
            ) from exc
        llm_kwargs["groq_api_key"] = GROQ_API_KEY
        return chat_groq_cls(**llm_kwargs)

    llm_kwargs["openai_api_key"] = OPENAI_API_KEY
    return ChatOpenAI(**llm_kwargs)


def get_default_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: Optional[bool] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Create a configured chat model using provider/model from settings."""
    resolved_provider = (provider or settings.llm.get("provider", "openai")).lower()
    resolved_model = _resolve_model(resolved_provider, model)
    resolved_temperature = settings.llm.get("temperature", 0.0) if temperature is None else temperature
    resolved_streaming = settings.llm.get("streaming", False) if streaming is None else streaming
    resolved_max_tokens = settings.llm.get("max_tokens", 2000) if max_tokens is None else max_tokens

    return _build_llm(
        provider=resolved_provider,
        model=resolved_model,
        temperature=resolved_temperature,
        streaming=resolved_streaming,
        max_tokens=resolved_max_tokens,
        **kwargs,
    )
