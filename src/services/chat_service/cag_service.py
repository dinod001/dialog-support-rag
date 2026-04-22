"""
CAG (Cache-Augmented Generation) service combining caching with CRAG.

Pipeline:
    Query --> Semantic Cache (Qdrant cag_cache KNN-1)
          --> HIT? Return instantly (0ms, $0)
          --> MISS? --> CRAGService (self-correcting retrieval)
                    --> Cache the result for future hits
                    --> Return answer

Logging:
    Uses the shared project logger with output written to ``logs/cag_service.log``.
"""

from typing import Any, Dict, List
import time
from pathlib import Path
import sys

# Allow direct execution: python src/services/chat_service/cag_service.py
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from infrastructure.log import get_logger
from services.chat_service.cag_cache import CAGCache
from services.chat_service.crag_service import CRAGService
from services.chat_service.conversation_memory import get_conversation_memory
from services.chat_service.rag_service import QdrantRetriever
from infrastructure.llm import get_default_embeddings, get_default_llm
from infrastructure.observability import observe, update_current_observation, flush

logger = get_logger(__name__, "cag_service.log")


class CAGService:
    """
    Cache-Augmented Generation backed by Corrective RAG.

    Layer 1: Semantic cache (Qdrant cag_cache) -- instant, $0
    Layer 2: CRAG (confidence-gated retrieval) -- self-correcting
    """

    def __init__(self, crag_service: CRAGService, cache: CAGCache):
        self.crag_service = crag_service
        self.cache = cache

    @observe(name="cag_generate")
    def generate(
        self,
        query: str,
        use_cache: bool = True,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Generate answer with CAG + CRAG pipeline.

        1. Check semantic cache (cosine >= 0.90 = HIT)
        2. On miss: run CRAGService (confidence-gated retrieval)
        3. Cache the result for future semantic hits
        """
        start = time.time()
        memory = get_conversation_memory(session_id)
        history = memory.load_memory_variables({}).get("history", "")

        if use_cache:
            cached = self.cache.get(query)
            if cached:
                latency_ms = int((time.time() - start) * 1000)
                logger.info(
                    "CAG cache HIT (score=%.3f) for: %s",
                    cached.get("score", 0),
                    query[:60],
                )
                update_current_observation(
                    input=query,
                    output=cached["answer"][:500],
                    metadata={
                        "cache_hit": True,
                        "cache_score": cached.get("score", 0),
                        "latency_ms": latency_ms,
                    },
                )
                return {
                    "answer": cached["answer"],
                    "evidence_urls": cached.get("evidence_urls", []),
                    "cache_hit": True,
                    "cache_score": cached.get("score", 0),
                    "generation_time": 0.0,
                }

        # Cache miss -- run CRAG (self-correcting retrieval)
            crag_result = self.crag_service.generate(query, verbose=False, chat_history=history)

        answer = crag_result.get("answer", "")
        evidence_urls = crag_result.get("evidence_urls", [])

        result: Dict[str, Any] = {
            "answer": answer,
            "evidence_urls": evidence_urls,
            "cache_hit": False,
            "confidence_initial": crag_result.get("confidence_initial", 0),
            "confidence_final": crag_result.get("confidence_final", 0),
            "correction_applied": crag_result.get("correction_applied", False),
            "generation_time": crag_result.get("generation_time", 0),
            "num_docs": crag_result.get("docs_used", 0),
        }

        if use_cache and answer:
            self.cache.set(query, {"answer": answer, "evidence_urls": evidence_urls})
            logger.info("CAG cache MISS -> cached for: %s", query[:60])

        if answer:
            memory.save_context({"question": query}, {"answer": answer})

        latency_ms = int((time.time() - start) * 1000)
        update_current_observation(
            input=query,
            output=answer[:500] if answer else "No results",
            metadata={
                "cache_hit": False,
                "latency_ms": latency_ms,
                "correction_applied": result["correction_applied"],
                "confidence_final": result["confidence_final"],
            },
        )

        return result

    def warm_cache(self, queries: List[str]) -> int:
        """Pre-populate cache with common queries via CRAG pipeline."""
        cached_count = 0
        for query in queries:
            if query not in self.cache:
                self.generate(query, use_cache=True)
                cached_count += 1
        return cached_count

    def cache_stats(self) -> Dict[str, Any]:
        return self.cache.stats()

    def clear_cache(self) -> None:
        self.cache.clear()


__all__ = ["CAGService"]

if __name__ == "__main__":
    try:
        embedder = get_default_embeddings()
        llm = get_default_llm()
        retriever = QdrantRetriever(embedder=embedder)
        crag = CRAGService(retriever=retriever, llm=llm)
        cache = CAGCache(embedder=embedder)
        service = CAGService(crag_service=crag, cache=cache)
        result = service.generate("what is Dialog ")
        logger.info("CAG answer preview: %s", result.get("answer", ""))
    except Exception as exc:
        logger.error("Failed to run CAGService main: %s", exc)
    finally:
        flush()
