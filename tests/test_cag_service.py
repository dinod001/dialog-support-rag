from pathlib import Path
import sys
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.chat_service.cag_cache import CAGCache
from services.chat_service.cag_service import CAGService
from services.chat_service.crag_service import CRAGService


class StaticRetriever(BaseRetriever):
    """Retriever that always returns the same evidence set for tests."""

    docs: List[Document]

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self.docs


def _build_service(cache_hit: Dict[str, Any] | None = None) -> tuple[CAGService, Mock]:
    docs = [
        Document(
            page_content="Dialog data package activation can be done in the self-care app.",
            metadata={
                "url": "https://www.dialog.lk/support/data-packages",
                "title": "Data Packages",
                "strategy": "parent_child",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="For SIM replacement and blocked SIM issues, visit a Dialog service center.",
            metadata={
                "url": "https://www.dialog.lk/support/sim-services",
                "title": "SIM Services",
                "strategy": "parent_child",
                "chunk_index": 1,
            },
        ),
    ]

    retriever = StaticRetriever(docs=docs)
    llm = RunnableLambda(
        lambda _prompt: "You can use the Dialog app or customer support. [https://www.dialog.lk/support/data-packages]"
    )
    crag_service = CRAGService(retriever=retriever, llm=llm, initial_k=2, expanded_k=3)

    cache = Mock(spec=CAGCache)
    cache.get.return_value = cache_hit
    cache.set = Mock()

    return CAGService(crag_service=crag_service, cache=cache), cache


def test_cag_generate_cache_miss_runs_real_crag_path() -> None:
    service, cache = _build_service(cache_hit=None)

    query = "How can I activate roaming on my Dialog number?"
    result = service.generate(query, use_cache=True)

    assert result["cache_hit"] is False
    assert result["answer"] != ""
    assert result["num_docs"] > 0
    cache.get.assert_called_once_with(query)
    cache.set.assert_called_once()


def test_cag_generate_cache_hit_skips_crag_generate() -> None:
    cached = {
        "answer": "Check via Dialog app quick actions.",
        "evidence_urls": ["https://www.dialog.lk/support"],
        "score": 0.95,
    }
    service, cache = _build_service(cache_hit=cached)
    query = "How do I check my account balance?"

    with patch.object(service.crag_service, "generate", wraps=service.crag_service.generate) as crag_spy:
        result = service.generate(query, use_cache=True)

    assert result["cache_hit"] is True
    assert result["answer"] == cached["answer"]
    assert crag_spy.call_count == 0
    cache.set.assert_not_called()


@pytest.mark.parametrize(
    "query",
    [
        "How can I buy a Dialog data package?",
        "My SIM is blocked, what should I do?",
        "How can I contact Dialog customer care?",
    ],
)
def test_cag_generate_with_sample_questions_uses_real_services(query: str) -> None:
    service, cache = _build_service(cache_hit=None)

    result = service.generate(query, use_cache=True)

    assert result["cache_hit"] is False
    assert result["answer"] != ""
    assert result["confidence_final"] >= 0.0
    assert isinstance(result["evidence_urls"], list)
    cache.set.assert_called_once()
