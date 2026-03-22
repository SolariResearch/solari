#!/usr/bin/env python3
"""
Solari CLI — Give your AI a brain, not just hands.

Usage:
    solari ingest --url URL --mind NAME
    solari ingest --pdf PATH --mind NAME
    solari query "your question" [--minds NAME1,NAME2] [--top N]
    solari agent "your question" [--provider anthropic|openai|ollama]
    solari dream --minds NAME1,NAME2 [--cycles N]
    solari minds                     # list available minds
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
            # Count entries
            meta_path = os.path.join(path, "metadata.json")
            entries = "?"
            if os.path.exists(meta_path):
                import json
                try:
                    meta = json.load(open(meta_path))
                    entries = meta.get("count", meta.get("total_entries", "?"))
                except Exception:
                    pass
            minds.append((d, entries))

    if not minds:
        print(f"No minds found in {minds_dir}")
        print("Use 'solari ingest' to create your first mind.")
        return

    print(f"\n  Available minds ({len(minds)}):\n")
    for name, count in minds:
        print(f"    {name:<30} {count} entries")
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
