"""
vibe_search tool -- semantic search over song vibe descriptions using ChromaDB.

The agent calls this when the user describes a situation or feeling in natural
language rather than specifying genre/mood attributes directly.

The ChromaDB index is built once at startup from songs_enriched.json using
local sentence-transformers embeddings (all-MiniLM-L6-v2, no API key needed).
"""

import json
import os
from typing import Optional

from langchain_core.tools import tool

ENRICHED_PATH = os.getenv("ENRICHED_PATH", "backend/data/songs_enriched.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "backend/data/chroma_db")
COLLECTION_NAME = "song_vibes"

_chroma_client = None
_collection = None


def _get_collection():
    """Lazily initialise ChromaDB collection, building the index if needed."""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        raise RuntimeError("chromadb is not installed. Run: pip install chromadb sentence-transformers")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    existing = [c.name for c in _chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        _collection = _chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
        return _collection

    _collection = _chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    with open(ENRICHED_PATH, encoding="utf-8") as f:
        songs = json.load(f)

    _collection.add(
        ids=[str(s["id"]) for s in songs],
        documents=[s["vibe"] for s in songs],
        metadatas=[{"title": s["title"], "artist": s["artist"], "song_id": s["id"]} for s in songs],
    )

    return _collection


def build_vibe_index() -> None:
    """Call this at app startup to pre-build the ChromaDB index."""
    _get_collection()


@tool
def vibe_search(
    query: str,
    exclude_ids: Optional[list[int]] = None,
    limit: int = 8,
) -> dict:
    """
    Search for songs by semantic similarity to a natural language vibe description.

    Use this tool when the user describes a feeling, situation, or atmosphere
    rather than specific genre or mood labels. For example: 'studying at 2am',
    'driving alone on a highway', 'rainy day coffee shop'.

    Args:
        query: Natural language description of the desired vibe or situation.
        exclude_ids: List of song IDs to exclude from results.
        limit: Maximum number of results to return (default 8).

    Returns:
        A dict with 'songs' (list of matches with similarity scores) and
        'query_used' so the agent knows what was searched.
    """
    try:
        collection = _get_collection()
    except Exception as e:
        return {
            "songs": [],
            "query_used": query,
            "error": str(e),
            "fallback_hint": "Use catalog_search with structured filters instead.",
        }

    exclude = set(exclude_ids or [])
    where_filter = None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(limit + len(exclude), 20),
        )
    except Exception as e:
        return {
            "songs": [],
            "query_used": query,
            "error": f"ChromaDB query failed: {e}",
            "fallback_hint": "Use catalog_search with structured filters instead.",
        }

    songs = []
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for song_id_str, distance, meta, doc in zip(ids, distances, metadatas, documents):
        song_id = int(song_id_str)
        if song_id in exclude:
            continue
        similarity = round(1 - distance, 4)
        songs.append({
            "id": song_id,
            "title": meta["title"],
            "artist": meta["artist"],
            "similarity": similarity,
            "vibe_description": doc,
        })
        if len(songs) >= limit:
            break

    return {
        "songs": songs,
        "query_used": query,
        "total_found": len(songs),
    }
