"""
Qdrant ingestion pipeline — load, chunk, embed, upsert.

This module contains the core service logic for ingesting documents
into Qdrant Cloud.  Scripts (``scripts/ingest_to_qdrant.py``) and CLI
commands should call :func:`run_ingest` rather than duplicating the
pipeline steps.
"""

import time
from pathlib import Path
from typing import Any, Dict, List
import sys

# Allow direct execution: python src/services/ingest_service/pipeline.py
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from infrastructure.log import get_logger
from infrastructure.llm.embeddings import get_default_embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from infrastructure.db.qdrant_client import (
    ensure_collection,
    delete_collection,
    upsert_chunks,
    collection_info,
    collection_exists,
)

from services.ingest_service.chunkers import parent_child_chunk
from infrastructure.config import settings

QDRANT_COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME or "dialog_memory"
EMBEDDING_BATCH_SIZE = settings.embedding.get("batch_size", 100)
logger = get_logger(__name__, "ingest_pipeline.log")


# =====================================================================
# Document loaders ( PDF + Text)
# =====================================================================

def _resolve_data_dir() -> Path:
    data_dir = settings.paths.get("data_dir", "data")
    return Path(settings.project_root) / data_dir


def _normalize_documents(raw_docs: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for doc in raw_docs:
        if isinstance(doc, dict):
            content = doc.get("content", "")
            if not content.strip():
                continue
            normalized.append(
                {
                    "content": content,
                    "url": doc.get("url") or doc.get("source") or "unknown",
                    "title": doc.get("title") or "Untitled",
                }
            )
            continue

        page_content = getattr(doc, "page_content", "")
        if not page_content or not str(page_content).strip():
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "unknown")
        title = metadata.get("title") or Path(str(source)).name or "Untitled"
        normalized.append(
            {
                "content": str(page_content),
                "url": str(source),
                "title": str(title),
            }
        )

    return normalized


def load_pdf_docs(source_path: Path) -> List[Dict[str, Any]]:
    if not source_path.exists():
        raise FileNotFoundError(f"PDF source path does not exist: {source_path}")

    pdf_files = [source_path] if source_path.is_file() else sorted(source_path.glob("*.pdf"))
    raw_docs: List[Any] = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(file_path=str(pdf_file))
        raw_docs.extend(loader.load())

    return _normalize_documents(raw_docs)


def load_text_docs(source_path: Path) -> List[Dict[str, Any]]:
    if not source_path.exists():
        raise FileNotFoundError(f"Text source path does not exist: {source_path}")

    if source_path.is_file():
        text_files = [source_path]
    else:
        text_files = sorted(source_path.glob("*.txt")) + sorted(source_path.glob("*.md"))

    raw_docs: List[Any] = []
    for text_file in text_files:
        loader = TextLoader(file_path=str(text_file), encoding="utf-8")
        raw_docs.extend(loader.load())

    return _normalize_documents(raw_docs)


# =====================================================================
# Embedding helper
# =====================================================================


def embed_texts(
    texts: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """Embed a list of texts using the configured embedding model."""
    embedder = get_default_embeddings(chunk_size=batch_size)
    all_embeddings: List[List[float]] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(
            "Embedding batch %s/%s (%s texts)...",
            batch_num,
            total_batches,
            len(batch),
        )
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# =====================================================================
# Parent-child helpers
# =====================================================================


def _build_parent_lookup(parents: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build a mapping from parent_id → parent text."""
    return {p["parent_id"]: p["text"] for p in parents}


def _enrich_children_with_parent_text(
    children: List[Dict[str, Any]],
    parent_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Attach ``parent_text`` to each child chunk for richer LLM context."""
    for child in children:
        pid = child.get("parent_id", "")
        child["parent_text"] = parent_lookup.get(pid, child["text"])
    return children


# =====================================================================
# Core pipeline
# =====================================================================

LOADER_MAP = {
    "pdf": load_pdf_docs,
    "text": load_text_docs,
}

def run_ingest(
    source: str = "pdf",
    source_path: Path | None = None,
    strategy: str = "parent_child",
    recreate: bool = False,
) -> int:
    """
    End-to-end ingestion pipeline.

    Args:
        source: One of ``pdf`` or ``text``.
        source_path: Optional file or directory path. Defaults to configured data dir.
        strategy: Chunking strategy. Currently only ``parent_child`` is implemented.
        recreate: If ``True``, drop and recreate the Qdrant collection first.

    Returns:
        Number of points upserted.

    Raises:
        ValueError: If *source* or *strategy* is unknown.
        FileNotFoundError: If the source directory does not exist.
    """
    logger.info("=" * 70)
    logger.info("🚀 QDRANT INGESTION PIPELINE")
    logger.info("=" * 70)

    # ── 1. Load documents ────────────────────────────────────
    loader = LOADER_MAP.get(source)
    if loader is None:
        raise ValueError(
            f"Unknown source: {source}. Choose from {list(LOADER_MAP.keys())}"
        )

    if strategy != "parent_child":
        raise ValueError(
            "Unsupported strategy: "
            f"{strategy}. Currently supported: ['parent_child']"
        )

    resolved_source_path = source_path or _resolve_data_dir()

    logger.info(f"\n📂 Loading documents (source={source}, path={resolved_source_path})...")
    docs = loader(resolved_source_path)
    if not docs:
        raise ValueError("No documents loaded. Nothing to ingest.")

    # ── 2. Chunk ─────────────────────────────────────────────

    children, parents = parent_child_chunk(docs)
    logger.info(f"   → {len(children)} child chunks, {len(parents)} parent chunks")
    parent_lookup = _build_parent_lookup(parents)
    chunks = _enrich_children_with_parent_text(children, parent_lookup)
    logger.info("   → Each child enriched with parent_text for richer LLM context")

    if not chunks:
        raise ValueError("No chunks produced. Check source documents and chunking settings.")

    # ── 3. Embed ─────────────────────────────────────────────
    logger.info(f"\n🔢 Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = embed_texts(texts)
    embed_secs = time.time() - t0
    logger.info(f"   → Embedding done in {embed_secs:.1f}s")

    if not embeddings or not embeddings[0]:
        raise ValueError("No embeddings produced. Check embedding provider configuration.")

    embedding_dim = len(embeddings[0])
    if any(len(vec) != embedding_dim for vec in embeddings):
        raise ValueError("Inconsistent embedding dimensions returned by provider.")
    logger.info(f"   → Embedding dimension detected: {embedding_dim}")

    # ── 4. Create / recreate collection ──────────────────────
    if recreate:
        logger.info(f"\n🗑️  Recreating collection '{QDRANT_COLLECTION_NAME}'...")
        try:
            delete_collection()
        except Exception:
            pass  # collection may not exist yet

    if not recreate and collection_exists():
        info = collection_info()
        existing_dim = int(info.get("vector_size", 0))
        if existing_dim != embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch with existing collection: "
                f"embeddings={embedding_dim}, collection={existing_dim}. "
                "Set recreate=True (or align embedding.vector_size) and run ingestion again."
            )

    ensure_collection(vector_size=embedding_dim)

    # ── 5. Upsert ────────────────────────────────────────────
    logger.info(f"\n⬆️  Upserting {len(chunks)} points into Qdrant...")
    t0 = time.time()
    n = upsert_chunks(chunks, embeddings)
    upsert_secs = time.time() - t0
    logger.info(f"   → Upserted {n} points in {upsert_secs:.1f}s")

    # ── 6. Verify ────────────────────────────────────────────
    logger.info("\n📊 Collection info:")
    info = collection_info()
    for k, v in info.items():
        logger.info(f"   {k}: {v}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ INGESTION COMPLETE")
    logger.info(f"   Source: {source}")
    logger.info(f"   Strategy: {strategy}")
    logger.info(f"   Chunks indexed: {n}")
    if strategy == "parent_child":
        logger.info("   Parent context: Stored in payload for richer LLM generation")
    logger.info("=" * 70)

    return n


if __name__ == "__main__":
    run_ingest(source="pdf", source_path=Path("data"), strategy="parent_child", recreate=True)