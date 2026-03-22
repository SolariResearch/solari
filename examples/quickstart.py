#!/usr/bin/env python3
"""Quickstart — Ingest a Wikipedia article and query it in under 30 seconds.

Usage:
    python examples/quickstart.py

No configuration needed. Creates a local `./minds/` directory with your
first knowledge index, then runs a semantic search against it.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solari.ingest import wikipedia_article, ingest_into_mind
from solari.query import query
from pathlib import Path

MINDS_DIR = "./minds"

# 1. Fetch a Wikipedia article
print("Fetching 'Transformer (deep learning)' from Wikipedia...")
title, text = wikipedia_article("Transformer (deep learning)")
if not text:
    print("Could not fetch article. Check your network connection.")
    sys.exit(1)
print(f"  Got {len(text):,} characters from '{title}'")

# 2. Chunk, embed, and store it as a searchable mind
count = ingest_into_mind("transformers", [text], source="wikipedia", minds_dir=MINDS_DIR)
print(f"  Indexed {count} new entries into the 'transformers' mind")

# 3. Query the mind with a natural-language question
results = query("How does self-attention work?", minds_dir=Path(MINDS_DIR), top_k=3)
print(f"\nTop {len(results)} results for 'How does self-attention work?':\n")
for i, r in enumerate(results, 1):
    print(f"  [{i}] (similarity={r['similarity']:.3f})")
    print(f"      {r['content'][:200]}...")
    print()
