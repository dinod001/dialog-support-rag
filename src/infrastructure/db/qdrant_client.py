from infrastructure.log import get_logger
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from infrastructure.config import settings


QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_URL = settings.QDRANT_URL
QDRANT_COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME or "dialogue_memory"
EMBEDDING_DIM = int(
    settings.embedding.get(
        "vector_size",
        1536 if settings.embedding.get("tier", "small") == "small" else 3072,
    )
)

logger = get_logger(__name__, "qdrant_client.log")
# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Return a singleton QdrantClient connected to Qdrant Cloud.

    Requires QDRANT_URL and QDRANT_API_KEY in .env.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL is not set.  Add it to your .env file.\n"
            "Example: QDRANT_URL=https://xxxxx.us-east.aws.cloud.qdrant.io"
        )
    if not QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_API_KEY is not set.  Add it to your .env file."
        )

    _qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )
    logger.info("Connected to Qdrant Cloud at %s", QDRANT_URL)
    return _qdrant_client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


def ensure_collection(
    collection_name: str = QDRANT_COLLECTION_NAME,
    vector_size: int = EMBEDDING_DIM,
    distance: Distance = Distance.COSINE,
    on_disk: bool = True,
) -> None:
    """
    Create the Qdrant collection if it does not exist.

    Safe to call repeatedly (idempotent).
    """
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info("Collection '%s' already exists; skipping creation.", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
            on_disk=on_disk,
        ),
    )
    logger.info(
        "Created Qdrant collection '%s' (dim=%s, distance=%s)",
        collection_name,
        vector_size,
        distance.name,
    )


def delete_collection(collection_name: str = QDRANT_COLLECTION_NAME) -> None:
    """Drop the entire collection (destructive)."""
    client = get_qdrant_client()
    client.delete_collection(collection_name)
    logger.info("Deleted Qdrant collection '%s'", collection_name)


def collection_info(collection_name: str = QDRANT_COLLECTION_NAME) -> Dict[str, Any]:
    """Return collection stats (point count, vector size, etc.)."""
    client = get_qdrant_client()
    info = client.get_collection(collection_name)
    return {
        "name": collection_name,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "vector_size": info.config.params.vectors.size,  # type: ignore[union-attr]
        "distance": info.config.params.vectors.distance.name,  # type: ignore[union-attr]
        "status": info.status.name,
    }


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: str = QDRANT_COLLECTION_NAME,
    batch_size: int = 100,
) -> int:
    """
    Upsert document chunks (with embeddings) into Qdrant.

    Each chunk dict is expected to contain at least:
        - text (str): The chunk content.
        - url (str): Source URL.
        - title (str): Source document title.
        - strategy (str): Chunking strategy used.
        - chunk_index (int): Position in the original document.

    Any extra keys are stored as payload metadata.

    Args:
        chunks: List of chunk dictionaries.
        embeddings: Parallel list of embedding vectors.
        collection_name: Target collection.
        batch_size: Points per upsert call.

    Returns:
        Number of points upserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

    client = get_qdrant_client()
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeds = embeddings[i : i + batch_size]

        points = []
        for chunk, vec in zip(batch_chunks, batch_embeds):
            point_id = str(uuid.uuid4())
            payload = {
                "chunk_text": chunk.get("text", ""),
                "url": chunk.get("url", ""),
                "title": chunk.get("title", ""),
                "strategy": chunk.get("strategy", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
            }
            # Include any extra metadata
            for k, v in chunk.items():
                if k not in ("text", "url", "title", "strategy", "chunk_index"):
                    payload[k] = v

            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
        logger.debug("Upserted batch %s-%s (%s points)", i, i + len(points), len(points))

    logger.info("Upserted %s points into '%s'", total, collection_name)
    return total


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_chunks(
    query_vector: List[float],
    top_k: int = 4,
    score_threshold: float = 0.0,
    collection_name: str = QDRANT_COLLECTION_NAME,
    strategy_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search over the RAG knowledge base.

    Args:
        query_vector: Embedded query vector.
        top_k: Number of results to return.
        score_threshold: Minimum cosine similarity (0–1).
        collection_name: Collection to search.
        strategy_filter: Optional — restrict to a chunking strategy.

    Returns:
        List of dicts with keys:
            chunk_text, url, title, strategy, chunk_index, score
    """
    client = get_qdrant_client()

    query_filter = None
    if strategy_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="strategy",
                    match=MatchValue(value=strategy_filter),
                )
            ]
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        score_threshold=score_threshold,
    )

    results = []
    for hit in response.points:
        payload = hit.payload or {}
        result = {
            "chunk_text": payload.get("chunk_text", ""),
            "url": payload.get("url", ""),
            "title": payload.get("title", ""),
            "strategy": payload.get("strategy", "unknown"),
            "chunk_index": payload.get("chunk_index", 0),
            "score": hit.score,
        }
        # Parent-child: include parent text for richer LLM context
        if "parent_text" in payload:
            result["parent_text"] = payload["parent_text"]
        if "parent_id" in payload:
            result["parent_id"] = payload["parent_id"]
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Convenience — count
# ---------------------------------------------------------------------------


def count_points(collection_name: str = QDRANT_COLLECTION_NAME) -> int:
    """Return the number of points in the collection."""
    client = get_qdrant_client()
    info = client.get_collection(collection_name)
    return info.points_count or 0


def collection_exists(collection_name: str = QDRANT_COLLECTION_NAME) -> bool:
    """Check whether *collection_name* exists in Qdrant."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    return collection_name in existing

