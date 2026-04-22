"""
Chunking helpers for ingestion.

This module currently provides:
- token counting via tiktoken
- parent-child chunking for retrieval with rich context

Chunk sizes are loaded from centralized settings.
"""

from typing import List, Dict, Any, Tuple
import tiktoken
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from infrastructure.config import settings

PARENT_CHUNK_SIZE = settings.chunking.get("parent_child", {}).get("parent_size", 1200)
CHILD_CHUNK_SIZE = settings.chunking.get("parent_child", {}).get("child_size", 250)
CHILD_OVERLAP = settings.chunking.get("parent_child", {}).get("child_overlap", 50)


# ============================================================================
# Utility Functions
# ============================================================================

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# ============================================================================
# Parent-Child (Two-Tier) Chunking
# ============================================================================

def parent_child_chunk(documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build child chunks linked to larger parent chunks.

    The function splits each document into large parent chunks and then
    creates smaller children inside each parent. Child records include a
    parent_id so retrieval can return precise hits with broader context.

    Expected document keys: content, url, title.

    Returns:
        Tuple of (children_chunks, parent_chunks).
    """
    parent_chunks = []
    child_chunks = []
    parent_idx = 0
    child_idx = 0
    
    # Character approximations
    parent_size_chars = PARENT_CHUNK_SIZE * 4
    child_size_chars = CHILD_CHUNK_SIZE * 4
    child_overlap_chars = CHILD_OVERLAP * 4
    
    # Parent splitter
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size_chars,
        chunk_overlap=200,  # Small overlap between parents
        length_function=len
    )
    
    # Child splitter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size_chars,
        chunk_overlap=child_overlap_chars,
        length_function=len
    )
    
    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Create parent chunks
        parent_texts = parent_splitter.split_text(content)
        
        for parent_text in parent_texts:
            if not parent_text.strip():
                continue
            
            parent_id = f"{url}::parent::{parent_idx}"
            
            # Store parent
            parent_chunks.append({
                "parent_id": parent_id,
                "url": url,
                "title": title,
                "text": parent_text.strip(),
                "strategy": "parent",
                "chunk_index": parent_idx,
                "token_count": count_tokens(parent_text)
            })
            
            # Create children within this parent
            child_texts = child_splitter.split_text(parent_text)
            
            for child_text in child_texts:
                if child_text.strip():
                    child_chunks.append({
                        "child_id": f"{parent_id}::child::{child_idx}",
                        "parent_id": parent_id,
                        "url": url,
                        "title": title,
                        "text": child_text.strip(),
                        "strategy": "child",
                        "chunk_index": child_idx,
                        "token_count": count_tokens(child_text)
                    })
                    child_idx += 1
            
            parent_idx += 1
    
    return child_chunks, parent_chunks



