import json
import time
import uuid
from typing import Any, Dict, List, Optional

from infrastructure.config import settings
from infrastructure.log import get_logger


# ── Defaults ─────────────────────────────────────────────────

_DEFAULT_COLLECTION = "dialog_cache"
logger = get_logger(__name__, "cag_cache.log")


class CAGCache:
    """
    Qdrant-backed semantic cache for pre-computed RAG responses.

    Each point stores:
        vector   → query embedding (float32)
        payload  → query, answer, evidence_urls, ts

    Lookup is a KNN-1 search; a hit is declared when
    ``cosine_similarity ≥ similarity_threshold``.

    Usage::

        cache = CAGCache(embedder=embedder)

        # Semantic lookup
        cached = cache.get("How many leave days?")
        if cached:
            return cached["answer"]

        # Store new entry (semantically indexed)
        cache.set("What is the leave policy?", {"answer": "...", "evidence_urls": [...]})
    """

    def __init__(
        self,
        embedder: Any,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Initialise the semantic CAG cache.

        Args:
            embedder: Object with ``embed_query(text) -> List[float]``.
            collection_name: Qdrant collection for CAG (default: ``cag_cache``).
            dim: Embedding dimension (auto-detected from config if None).
            similarity_threshold: Min cosine similarity for a hit (0.90–0.95).
            ttl_seconds: Entries older than this are ignored; 0 → no expiry.
        """
        self.embedder = embedder
        self.collection_name = collection_name or settings.cag.get("collection_name", "cag_cache")
        self.dim = dim or int(settings.embedding.get("vector_size", 1536))
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else float(settings.cag.get("similarity_threshold", 0.90))
        )
        self.ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else int(settings.cag.get("cache_ttl", 86400))
        )

        # Ensure the Qdrant collection exists
        self._available = False
        try:
            from infrastructure.db.qdrant_client import (
                get_qdrant_client,
                collection_exists,
            )

            self._client = get_qdrant_client()
            if not collection_exists(self.collection_name):
                self._create_collection()
            self._available = True
            logger.info(
                "✓ CAG cache ready (Qdrant collection='%s', dim=%s, threshold=%.2f)",
                self.collection_name,
                self.dim,
                self.similarity_threshold,
            )
        except Exception as exc:
            logger.warning(
                "CAG cache DISABLED — Qdrant unavailable: %s. "
                "All lookups will miss; every query runs full RAG.",
                exc,
            )

    # ── collection management ─────────────────────────────────

    def _create_collection(self) -> None:
        """Create the Qdrant collection for CAG cache."""
        from qdrant_client.http.models import Distance, VectorParams

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
                on_disk=False,  # keep in-memory for speed
            ),
        )
        logger.info(
            "Created CAG cache collection '%s' (dim=%s, COSINE)",
            self.collection_name,
            self.dim,
        )

    # ── public API ────────────────────────────────────────────

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Semantic cache lookup via KNN-1 search.

        Embeds *query*, searches the Qdrant HNSW index, and returns
        the cached response if ``cosine_similarity ≥ threshold``.

        Args:
            query: Natural-language question.

        Returns:
            Dict with ``query``, ``answer``, ``evidence_urls``, ``ts``,
            ``score`` — or ``None`` on miss.
        """
        if not self._available:
            return None

        # Embed query
        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("CAG embed failed: %s", exc)
            return None

        # KNN-1 search
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=1,
                score_threshold=self.similarity_threshold,
            )
        except Exception as exc:
            logger.warning("CAG cache GET error: %s", exc)
            return None

        if not response.points:
            return None

        hit = response.points[0]
        similarity = hit.score
        payload = hit.payload or {}

        # TTL filtering: skip entries older than ttl_seconds
        if self.ttl_seconds and self.ttl_seconds > 0:
            entry_ts = payload.get("ts", 0)
            if entry_ts and (time.time() - float(entry_ts)) > self.ttl_seconds:
                return None

        cached_query = payload.get("query", "")
        logger.info(
            "CAG cache HIT (sim=%.3f): '%s' -> matched '%s'",
            similarity,
            query[:50],
            cached_query[:50],
        )

        evidence_raw = payload.get("evidence_urls", "[]")
        try:
            evidence_urls = json.loads(evidence_raw) if isinstance(evidence_raw, str) else evidence_raw
        except (json.JSONDecodeError, TypeError):
            evidence_urls = []

        return {
            "query": cached_query,
            "answer": payload.get("answer", ""),
            "evidence_urls": evidence_urls,
            "ts": float(payload.get("ts", 0)),
            "score": similarity,
        }

    def set(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache a response, indexed by the query's embedding.

        Args:
            query: Original user query.
            response: Dict with ``answer`` and optionally ``evidence_urls``.
        """
        if not self._available:
            return

        # Embed query
        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("CAG embed failed on SET: %s", exc)
            return

        from qdrant_client.http.models import PointStruct

        point_id = str(uuid.uuid4())
        payload = {
            "query": query,
            "answer": response.get("answer", ""),
            "evidence_urls": json.dumps(response.get("evidence_urls", [])),
            "ts": time.time(),
        }

        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=query_vec, payload=payload)],
            )
            logger.debug("CAG cache SET: '%s' -> point=%s", query[:60], point_id)
        except Exception as exc:
            logger.warning("CAG cache SET error: %s", exc)

    def clear(self) -> None:
        """
        Drop and recreate the CAG cache collection.

        All cached entries are removed. The collection is recreated
        immediately so the cache is ready for new entries.
        """
        if not self._available:
            return

        try:
            self._client.delete_collection(self.collection_name)
            logger.info("Dropped CAG cache collection '%s'", self.collection_name)
        except Exception:
            pass  # Collection may not exist

        self._create_collection()
        logger.info("CAG cache cleared and collection recreated")

    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with ``total_cached``, ``backend``, ``collection``,
            ``similarity_threshold``, ``ttl_seconds``.
        """
        return {
            "total_cached": self._count(),
            "backend": "qdrant",
            "collection": self.collection_name,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl_seconds,
            "available": self._available,
        }
