#!/usr/bin/env python3
"""Standalone FAISS knowledge retrieval engine.

Loads FAISS vector indices from a minds directory, encodes a natural-language
query with sentence-transformers, and returns the top-k most relevant entries
ranked by cosine similarity.

Directory layout expected under --minds-dir (default: ./minds/):

    minds/
      physics/
        index.faiss
        metadata.json          # or metadata.json.gz
      chemistry/
        index.faiss
        metadata.json.gz
      ...

Each metadata file is a JSON list of objects.  Recognised fields per entry:
  - "content", "text", or "description" (used for display)
  - "category" (optional, shown in output)

Usage examples:
    python query.py "what is quantum entanglement" --minds-dir ./minds/ --top 5
    python query.py "query" --minds physics,chemistry --top 3
    python query.py --list --minds-dir ./minds/
    python query.py "query" --json
"""

import sys
import os
import json
import gzip
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any


DEFAULT_MINDS_DIR = Path("./minds/")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_minds(minds_dir: Path) -> List[tuple]:
    """List all minds with entry counts.

    Returns a list of (name, count) tuples sorted by count descending.
    """
    if not minds_dir.is_dir():
        print(f"Error: minds directory not found: {minds_dir}")
        return []

    minds: List[tuple] = []
    for d in sorted(minds_dir.iterdir()):
        if not d.is_dir():
            continue
        count = _count_entries(d)
        if count > 0:
            minds.append((d.name, count))

    minds.sort(key=lambda x: -x[1])
    total = sum(c for _, c in minds)
    print(f"Total: {total:,} entries across {len(minds)} minds\n")
    for name, count in minds:
        print(f"  {name}: {count:,}")
    return minds


def _count_entries(mind_dir: Path) -> int:
    """Return the number of metadata entries in a mind directory."""
    meta = _load_metadata(mind_dir)
    return len(meta)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _load_metadata(mind_dir: Path) -> list:
    """Load metadata from a mind directory (supports .json and .json.gz)."""
    meta_gz = mind_dir / "metadata.json.gz"
    meta_json = mind_dir / "metadata.json"

    meta: list = []
    if meta_gz.exists():
        with gzip.open(meta_gz, "rt", encoding="utf-8") as f:
            meta = json.load(f)
    elif meta_json.exists():
        with open(meta_json, encoding="utf-8") as f:
            meta = json.load(f)

    # Some indices store {"entries": [...]} instead of a bare list.
    if isinstance(meta, dict):
        meta = meta.get("entries", [])

    return meta


# ---------------------------------------------------------------------------
# Core query
# ---------------------------------------------------------------------------

def query(
    text: str,
    minds_dir: Path = DEFAULT_MINDS_DIR,
    mind_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Query FAISS indices and return the most relevant results.

    Parameters
    ----------
    text : str
        Natural-language query.
    minds_dir : Path
        Root directory containing per-mind subdirectories.
    mind_names : list[str] or None
        Specific minds to search.  If *None*, all available minds are searched.
    top_k : int
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Each dict has keys: content, similarity, mind, category, raw_score.
    """
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    if not minds_dir.is_dir():
        print(f"Error: minds directory not found: {minds_dir}", file=sys.stderr)
        return []

    # Encode the query
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    q_emb = encoder.encode([text], normalize_embeddings=True).astype("float32")

    # Resolve which minds to search
    if mind_names is None:
        mind_names = [
            d.name
            for d in sorted(minds_dir.iterdir())
            if d.is_dir() and (d / "index.faiss").exists()
        ]

    results: List[Dict[str, Any]] = []

    for mind_name in mind_names:
        mind_dir = minds_dir / mind_name
        index_path = mind_dir / "index.faiss"
        if not index_path.exists():
            continue

        index = faiss.read_index(str(index_path))
        if index.ntotal == 0:
            continue

        meta = _load_metadata(mind_dir)

        k = min(top_k * 2, index.ntotal)
        scores, indices = index.search(q_emb, k)

        is_ip = index.metric_type == 1  # METRIC_INNER_PRODUCT

        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(meta):
                continue

            # Normalise to a cosine-style similarity in [0, 1].
            if is_ip:
                # Inner-product on normalised vectors approximates cosine sim.
                # Quantised indices can occasionally exceed 1.0.
                similarity = 2.0 - float(score) if float(score) > 1.0 else float(score)
            else:
                # L2 distance: 0 means identical vectors.
                similarity = max(0.0, 1.0 - float(score) / 2.0)

            entry = meta[idx] if isinstance(meta[idx], dict) else {"content": str(meta[idx])}
            content = entry.get("content", entry.get("text", entry.get("description", "")))
            if not content:
                continue

            results.append({
                "content": content[:600],
                "similarity": round(similarity, 3),
                "mind": mind_name,
                "category": entry.get("category", ""),
                "raw_score": float(score),
            })

    # Rank by similarity (highest first) and deduplicate.
    results.sort(key=lambda x: x["similarity"], reverse=True)
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in results:
        key = r["content"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped[:top_k]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAISS knowledge retrieval engine — search vector minds from the command line.",
    )
    parser.add_argument("query", nargs="?", help="Natural-language search query")
    parser.add_argument(
        "--minds-dir",
        type=Path,
        default=DEFAULT_MINDS_DIR,
        help="Root directory containing mind subdirectories (default: ./minds/)",
    )
    parser.add_argument(
        "--minds",
        type=str,
        default=None,
        help="Comma-separated list of specific minds to search",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--list", action="store_true", help="List all minds with entry counts")
    parser.add_argument("--stats", action="store_true", help="Alias for --list")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output results as JSON")
    args = parser.parse_args()

    minds_dir: Path = args.minds_dir

    # --list / --stats mode
    if args.list or args.stats:
        list_minds(minds_dir)
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    mind_names = args.minds.split(",") if args.minds else None
    results = query(args.query, minds_dir=minds_dir, mind_names=mind_names, top_k=args.top)

    if not results:
        if args.json_output:
            print(json.dumps([]))
        else:
            print("No results found.")
        return

    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} [{r['mind']}] (sim={r['similarity']:.3f}) ---")
            print(r["content"])


if __name__ == "__main__":
    main()
