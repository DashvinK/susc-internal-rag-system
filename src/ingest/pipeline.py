"""Ingestion pipeline: sources -> parse -> clean -> chunk -> embed -> export.

Runs on YOUR laptop (full tooling). Writes the three small files the live
Vercel function loads:
    index/chunks.json       chunk text + metadata (source, section)
    index/vectors.npy       precomputed embeddings (offline model)
    index/bm25_corpus.json  tokenized corpus for rank_bm25
plus index/manifest.json (per-file content hashes) to drive INCREMENTAL refresh:
only changed/new files are re-parsed and re-embedded; unchanged files reuse
their existing chunks and vectors.

SAFETY: the default source is data/public ONLY. Confidential material in
data/private must never reach the committed index -- pass --source explicitly
(and a non-default --out) if you ever build a private, local-only index.

Usage:
    python -m src.ingest.pipeline                 # incremental, data/public -> index/
    python -m src.ingest.pipeline --full          # force full rebuild
    python -m src.ingest.pipeline --source data/private --out local_index/private
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src import config
from src.ingest import clean
from src.ingest.chunker import Chunk, chunk_pages
from src.ingest.parsers import SUPPORTED_SUFFIXES, parse_file
from src.retrieval import bm25 as bm25_mod


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Folder-level docs that explain the corpus but are not themselves corpus.
_IGNORE_NAMES = {"readme.md", "readme.txt", "readme"}


def _discover(source_dir: Path) -> dict[str, Path]:
    """Map source filename -> path for all supported files under source_dir."""
    found: dict[str, Path] = {}
    for path in sorted(source_dir.rglob("*")):
        if path.name.lower() in _IGNORE_NAMES:
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            if path.name in found:
                raise ValueError(
                    f"Duplicate filename '{path.name}'. Source filenames must be unique "
                    "because they are used as citation keys."
                )
            found[path.name] = path
    return found


def _chunks_for_file(path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    pages = parse_file(path)
    for page in pages:
        page.text = clean.clean_text(page.text)
    pages = [p for p in pages if p.text]
    return chunk_pages(pages, chunk_size, overlap)


def build_index(source_dir: Path, out_dir: Path, full: bool) -> dict:
    from src.retrieval import embedder_offline  # heavy import, offline only

    chunk_size = int(config.get("chunking.chunk_size", 2000))
    overlap = int(config.get("chunking.overlap", 200))

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.json"
    vectors_path = out_dir / "vectors.npy"
    bm25_path = out_dir / "bm25_corpus.json"
    manifest_path = out_dir / "manifest.json"

    current_files = _discover(source_dir)
    current_hashes = {name: _file_hash(p) for name, p in current_files.items()}

    # --- Load existing index for incremental reuse ---
    old_chunks: list[dict] = []
    old_vectors: np.ndarray | None = None
    old_manifest: dict[str, str] = {}
    if not full and chunks_path.exists() and vectors_path.exists() and manifest_path.exists():
        old_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        old_vectors = np.load(vectors_path)
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Group old chunks + their vector rows by source file.
    old_by_source: dict[str, tuple[list[dict], list[np.ndarray]]] = {}
    if old_vectors is not None and len(old_chunks) == old_vectors.shape[0]:
        for row, ch in enumerate(old_chunks):
            bucket = old_by_source.setdefault(ch["source"], ([], []))
            bucket[0].append(ch)
            bucket[1].append(old_vectors[row])

    reused, changed, added, removed = [], [], [], []
    for name in old_manifest:
        if name not in current_hashes:
            removed.append(name)

    final_chunks: list[dict] = []
    final_vectors: list[np.ndarray] = []

    for name in sorted(current_files):
        unchanged = old_manifest.get(name) == current_hashes[name] and name in old_by_source
        if unchanged:
            chs, vecs = old_by_source[name]
            final_chunks.extend(chs)
            final_vectors.extend(vecs)
            reused.append(name)
            continue

        (changed if name in old_manifest else added).append(name)
        new_chunks = _chunks_for_file(current_files[name], chunk_size, overlap)
        if not new_chunks:
            continue
        vecs = embedder_offline.embed_passages([c.text for c in new_chunks])
        final_chunks.extend(c.to_dict() for c in new_chunks)
        final_vectors.extend(vecs)

    if not final_chunks:
        raise SystemExit(
            f"No chunks produced from {source_dir}. Add documents (.pdf/.docx/.md/.txt) first."
        )

    vectors_array = np.vstack(final_vectors).astype(np.float32)
    bm25_corpus = [bm25_mod.tokenize(c["text"]) for c in final_chunks]

    # --- Write exports ---
    chunks_path.write_text(json.dumps(final_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(vectors_path, vectors_array)
    bm25_path.write_text(json.dumps(bm25_corpus, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(current_hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "n_files": len(current_files),
        "n_chunks": len(final_chunks),
        "vector_shape": list(vectors_array.shape),
        "reused": reused,
        "changed": changed,
        "added": added,
        "removed": removed,
        "full_rebuild": full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG index (offline).")
    parser.add_argument(
        "--source",
        default=str(config.resolve_path("paths.public_data")),
        help="Directory of source documents (default: data/public).",
    )
    parser.add_argument(
        "--out",
        default=str(config.resolve_path("paths.index_dir")),
        help="Output directory for the exported index (default: index/).",
    )
    parser.add_argument("--full", action="store_true", help="Force a full rebuild.")
    args = parser.parse_args()

    summary = build_index(Path(args.source), Path(args.out), args.full)

    print("\n=== Ingestion complete ===")
    print(f"  source:   {summary['source_dir']}")
    print(f"  output:   {summary['out_dir']}")
    print(f"  files:    {summary['n_files']}")
    print(f"  chunks:   {summary['n_chunks']}")
    print(f"  vectors:  {summary['vector_shape']}")
    print(
        f"  reused:   {len(summary['reused'])}  "
        f"added: {len(summary['added'])}  "
        f"changed: {len(summary['changed'])}  "
        f"removed: {len(summary['removed'])}"
    )
    if summary["added"]:
        print(f"    + {', '.join(summary['added'])}")
    if summary["changed"]:
        print(f"    ~ {', '.join(summary['changed'])}")
    if summary["removed"]:
        print(f"    - {', '.join(summary['removed'])} (dropped from index)")
    print("\nReminder: manually read index/chunks.json before committing -- it is the")
    print("one file that actually ships. Nothing confidential should appear in it.")


if __name__ == "__main__":
    main()
