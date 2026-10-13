"""ChromaDB persistent store -- LOCAL ONLY, for the ablation study.

Never imported by the live function (chromadb is far too heavy for Vercel).
Gives the Week-5 experiments a convenient persistent vector store to compare
configurations against. The production path uses plain NumPy instead.
"""

from __future__ import annotations

from src import config
from src.ingest.chunker import Chunk


def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(config.resolve_path("paths.local_index_dir")))


def build_collection(name: str, chunks: list[Chunk], vectors) -> None:
    """(Re)create a Chroma collection from chunks + precomputed vectors."""
    client = _client()
    try:
        client.delete_collection(name)
    except Exception:
        pass  # collection may not exist yet
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c.id for c in chunks],
        embeddings=[v.tolist() for v in vectors],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "section": c.section} for c in chunks],
    )


def query(name: str, query_vector, k: int) -> list[tuple[str, float]]:
    """Return ``(chunk_id, distance)`` for the top-k nearest chunks."""
    client = _client()
    collection = client.get_collection(name)
    res = collection.query(query_embeddings=[query_vector.tolist()], n_results=k)
    ids = res["ids"][0]
    distances = res["distances"][0]
    return list(zip(ids, distances, strict=False))
