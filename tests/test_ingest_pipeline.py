from pathlib import Path
import sys
import types
import importlib

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Fall back to local stubs only when optional dependencies are missing.
try:
    importlib.import_module("infrastructure.llm.embeddings")
except Exception:
    fake_embeddings_module = types.ModuleType("infrastructure.llm.embeddings")

    def _fake_get_default_embeddings(*_args: object, **_kwargs: object) -> object:
        class _FakeEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 3 for _ in texts]

        return _FakeEmbedder()

    fake_embeddings_module.get_default_embeddings = _fake_get_default_embeddings
    sys.modules["infrastructure.llm.embeddings"] = fake_embeddings_module

try:
    importlib.import_module("langchain_community.document_loaders")
except Exception:
    fake_loader_module = types.ModuleType("langchain_community.document_loaders")

    class _Doc:
        def __init__(self, page_content: str, metadata: dict[str, str]) -> None:
            self.page_content = page_content
            self.metadata = metadata

    class _StubTextLoader:
        def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
            self.file_path = file_path
            self.encoding = encoding

        def load(self) -> list[_Doc]:
            content = Path(self.file_path).read_text(encoding=self.encoding)
            return [_Doc(content, {"source": self.file_path, "title": Path(self.file_path).name})]

    class _StubPyPDFLoader:
        def __init__(self, file_path: str) -> None:
            self.file_path = file_path

        def load(self) -> list[_Doc]:
            return [_Doc("pdf stub content", {"source": self.file_path, "title": Path(self.file_path).name})]

    fake_loader_module.TextLoader = _StubTextLoader
    fake_loader_module.PyPDFLoader = _StubPyPDFLoader
    sys.modules["langchain_community.document_loaders"] = fake_loader_module

try:
    importlib.import_module("infrastructure.db.qdrant_client")
except Exception:
    fake_qdrant_module = types.ModuleType("infrastructure.db.qdrant_client")

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    def _fake_upsert(chunks: list[dict[str, str]], _embeddings: list[list[float]]) -> int:
        return len(chunks)

    def _fake_info() -> dict[str, int]:
        return {"points_count": 0}

    fake_qdrant_module.ensure_collection = _noop
    fake_qdrant_module.delete_collection = _noop
    fake_qdrant_module.upsert_chunks = _fake_upsert
    fake_qdrant_module.collection_info = _fake_info
    sys.modules["infrastructure.db.qdrant_client"] = fake_qdrant_module

try:
    importlib.import_module("services.ingest_service.chunkers")
except Exception:
    fake_chunkers_module = types.ModuleType("services.ingest_service.chunkers")

    def _fake_parent_child_chunk(documents: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        if not documents:
            return [], []
        parent_id = "test-parent-0"
        child = {
            "child_id": "test-child-0",
            "parent_id": parent_id,
            "url": documents[0].get("url", "unknown"),
            "title": documents[0].get("title", "Untitled"),
            "text": documents[0].get("content", ""),
        }
        parent = {
            "parent_id": parent_id,
            "text": documents[0].get("content", ""),
        }
        return [child], [parent]

    fake_chunkers_module.parent_child_chunk = _fake_parent_child_chunk
    sys.modules["services.ingest_service.chunkers"] = fake_chunkers_module

from services.ingest_service import pipeline


DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "document.pdf"
SAMPLE_TEXT_PATH = DATA_DIR / "sample_ingest_test.txt"


def test_load_pdf_docs_from_data_file() -> None:
    docs = pipeline.load_pdf_docs(PDF_PATH)

    assert isinstance(docs, list)
    assert len(docs) > 0
    assert "content" in docs[0]
    assert "url" in docs[0]
    assert "title" in docs[0]
    assert docs[0]["content"].strip() != ""


def test_load_text_docs_from_sample_file() -> None:
    docs = pipeline.load_text_docs(SAMPLE_TEXT_PATH)

    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0]["content"].strip() != ""
    assert docs[0]["url"].endswith("sample_ingest_test.txt")


def test_run_ingest_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_docs = [{"content": "hello world", "url": "u", "title": "t"}]
    children = [{"parent_id": "p1", "text": "child text", "url": "u", "title": "t"}]
    parents = [{"parent_id": "p1", "text": "parent text"}]

    calls: dict[str, object] = {}

    def fake_loader(_source_path: Path) -> list[dict[str, str]]:
        return loaded_docs

    def fake_parent_child_chunk(_docs: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        return children, parents

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        calls["embedded_texts"] = texts
        return [[0.1, 0.2, 0.3]]

    def fake_delete_collection() -> None:
        calls["deleted"] = True

    def fake_ensure_collection() -> None:
        calls["ensured"] = True

    def fake_upsert_chunks(chunks: list[dict[str, str]], embeddings: list[list[float]]) -> int:
        calls["upsert_chunks"] = chunks
        calls["upsert_embeddings"] = embeddings
        return len(chunks)

    def fake_collection_info() -> dict[str, int]:
        return {"points_count": 1}

    monkeypatch.setitem(pipeline.LOADER_MAP, "text", fake_loader)
    monkeypatch.setattr(pipeline, "parent_child_chunk", fake_parent_child_chunk)
    monkeypatch.setattr(pipeline, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(pipeline, "delete_collection", fake_delete_collection)
    monkeypatch.setattr(pipeline, "ensure_collection", fake_ensure_collection)
    monkeypatch.setattr(pipeline, "upsert_chunks", fake_upsert_chunks)
    monkeypatch.setattr(pipeline, "collection_info", fake_collection_info)

    indexed = pipeline.run_ingest(source="text", source_path=DATA_DIR, recreate=True)

    assert indexed == 1
    assert calls["embedded_texts"] == ["child text"]
    assert calls["deleted"] is True
    assert calls["ensured"] is True

    stored_chunks = calls["upsert_chunks"]
    assert isinstance(stored_chunks, list)
    assert stored_chunks[0]["parent_text"] == "parent text"
    assert calls["upsert_embeddings"] == [[0.1, 0.2, 0.3]]


def test_run_ingest_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        pipeline.run_ingest(source="csv")


def test_run_ingest_unsupported_strategy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        pipeline.LOADER_MAP,
        "text",
        lambda _source_path: [{"content": "x", "url": "u", "title": "t"}],
    )

    with pytest.raises(ValueError, match="Unsupported strategy"):
        pipeline.run_ingest(source="text", strategy="flat", source_path=DATA_DIR)


def test_run_ingest_no_documents_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(pipeline.LOADER_MAP, "text", lambda _source_path: [])

    with pytest.raises(ValueError, match="No documents loaded"):
        pipeline.run_ingest(source="text", source_path=DATA_DIR)
