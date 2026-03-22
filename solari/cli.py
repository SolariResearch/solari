#!/usr/bin/env python3
"""
Solari CLI — Give your AI a brain, not just hands.

Usage:
    solari ingest --url URL --mind NAME
    solari ingest --pdf PATH --mind NAME
    solari query "your question" [--minds NAME1,NAME2] [--top N]
    solari agent "your question" [--provider anthropic|openai|ollama]
    solari dream --minds NAME1,NAME2 [--cycles N]
    solari minds                     # list available minds + confidence
    solari prune --mind NAME         # remove low-confidence entries
"""

import argparse
import sys
import os


def cmd_ingest(args):
    """Route to ingest module."""
    from solari.ingest import main as ingest_main
    sys.argv = ["solari-ingest"] + args
    ingest_main()


def cmd_query(args):
    """Route to query module."""
    from solari.query import main as query_main
    sys.argv = ["solari-query"] + args
    query_main()


def cmd_agent(args):
    """Route to agent module."""
    from solari.agent import main as agent_main
    sys.argv = ["solari-agent"] + args
    agent_main()


def cmd_dream(args):
    """Route to dream module."""
    try:
        from solari.dream import main as dream_main
        sys.argv = ["solari-dream"] + args
        dream_main()
    except ImportError:
        print("Dream engine not yet available. Coming soon.")
        sys.exit(1)


def cmd_prune(args):
    """Prune low-confidence entries from a mind."""
    import argparse as ap
    parser = ap.ArgumentParser(description="Prune low-confidence entries from a mind.")
    parser.add_argument("--mind", required=True, help="Mind to prune")
    parser.add_argument("--minds-dir", default="./minds", help="Root mind directory")
    parser.add_argument(
        "--below", type=float, default=0.3,
        help="Remove entries with confidence below this value (default: 0.3)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pruned without doing it")
    parsed = parser.parse_args(args)

    import gzip
    import json
    from pathlib import Path

    mind_dir = Path(parsed.minds_dir) / parsed.mind
    meta_gz = mind_dir / "metadata.json.gz"
    meta_json = mind_dir / "metadata.json"
    index_path = mind_dir / "index.faiss"

    if not mind_dir.exists():
        print(f"Mind '{parsed.mind}' not found at {mind_dir}")
        sys.exit(1)

    # Load metadata
    meta = []
    if meta_gz.exists():
        with gzip.open(meta_gz, "rt") as f:
            meta = json.load(f)
    elif meta_json.exists():
        with open(meta_json) as f:
            meta = json.load(f)

    if not meta:
        print(f"Mind '{parsed.mind}' has no entries.")
        return

    keep = []
    prune = []
    for i, entry in enumerate(meta):
        conf = entry.get("confidence", 0.5)
        if conf < parsed.below:
            prune.append((i, conf, entry.get("content", "")[:60]))
        else:
            keep.append(i)

    print(f"\n  Mind: {parsed.mind}")
    print(f"  Total entries: {len(meta)}")
    print(f"  Below {parsed.below}: {len(prune)}")
    print(f"  Keeping: {len(keep)}")

    if prune:
        print(f"\n  Entries to prune:")
        for idx, conf, preview in prune[:20]:
            print(f"    [{idx}] conf={conf:.2f} | {preview}...")
        if len(prune) > 20:
            print(f"    ... and {len(prune) - 20} more")

    if parsed.dry_run or not prune:
        if not prune:
            print("\n  Nothing to prune.")
        return

    # Rebuild index without pruned entries
    import faiss
    import numpy as np

    if index_path.exists():
        old_index = faiss.read_index(str(index_path))
        vectors = old_index.reconstruct_n(0, old_index.ntotal)
        keep_vectors = np.array([vectors[i] for i in keep])
        new_index = faiss.IndexFlatIP(vectors.shape[1])
        if len(keep_vectors) > 0:
            new_index.add(keep_vectors)
        faiss.write_index(new_index, str(index_path))

    kept_meta = [meta[i] for i in keep]
    with gzip.open(meta_gz, "wt") as f:
        json.dump(kept_meta, f)

    print(f"\n  Pruned {len(prune)} entries. {len(keep)} remaining.")


def cmd_minds(args):
    """List available minds."""
    minds_dir = "./minds"
    for a in args:
        if a.startswith("--minds-dir"):
            if "=" in a:
                minds_dir = a.split("=", 1)[1]

    if not os.path.isdir(minds_dir):
        print(f"No minds directory found at {minds_dir}")
        print("Use 'solari ingest' to create your first mind.")
        return

    minds = []
    for d in sorted(os.listdir(minds_dir)):
        path = os.path.join(minds_dir, d)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "index.faiss")):
            meta_path = os.path.join(path, "metadata.json.gz")
            meta_plain = os.path.join(path, "metadata.json")
            entries = "?"
            avg_conf = None
            import gzip as _gz
            import json as _json
            try:
                if os.path.exists(meta_path):
                    with _gz.open(meta_path, "rt") as f:
                        data = _json.load(f)
                elif os.path.exists(meta_plain):
                    with open(meta_plain) as f:
                        data = _json.load(f)
                else:
                    data = []
                entries = len(data)
                confs = [e.get("confidence", 0.5) for e in data if isinstance(e, dict)]
                if confs:
                    avg_conf = sum(confs) / len(confs)
            except Exception:
                pass
            minds.append((d, entries, avg_conf))

    if not minds:
        print(f"No minds found in {minds_dir}")
        print("Use 'solari ingest' to create your first mind.")
        return

    print(f"\n  Available minds ({len(minds)}):\n")
    for name, count, conf in minds:
        conf_str = f"  avg_conf={conf:.2f}" if conf is not None else ""
        print(f"    {name:<30} {count} entries{conf_str}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    remaining = sys.argv[2:]

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "agent": cmd_agent,
        "dream": cmd_dream,
        "minds": cmd_minds,
        "list": cmd_minds,
        "prune": cmd_prune,
    }

    if command in ("--help", "-h", "help"):
        print(__doc__)
        sys.exit(0)

    if command not in commands:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(commands.keys())}")
        sys.exit(1)

    commands[command](remaining)


if __name__ == "__main__":
    main()
