#!/usr/bin/env python3
"""Dream Engine Demo — Discover hidden connections between unrelated knowledge domains.

Ingests 3 Wikipedia topics into separate minds, then runs the dream engine
to find cross-domain bridges and (optionally) synthesize novel insights.

Requirements:
    pip install faiss-cpu sentence-transformers numpy requests beautifulsoup4

    For REM-phase hypothesis generation (optional):
        Ollama must be running locally with a model pulled:
            ollama pull qwen2.5:7b
        If Ollama is not running, the script still discovers bridges (NREM only).

Usage:
    python examples/dream_demo.py
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solari.ingest import wikipedia_article, ingest_into_mind

MINDS_DIR = "./minds"

# ── Step 1: Build 3 separate knowledge minds ────────────────────────

topics = {
    "immunology": "Immune system",
    "distributed_systems": "Distributed computing",
    "game_theory": "Game theory",
}

print("=" * 60)
print("  DREAM ENGINE DEMO")
print("  Finding hidden structure across unrelated domains")
print("=" * 60)

for mind_name, wiki_title in topics.items():
    print(f"\n[ingest] Fetching '{wiki_title}' from Wikipedia...")
    title, text = wikipedia_article(wiki_title)
    if not text:
        print(f"  WARNING: Could not fetch '{wiki_title}'. Skipping.")
        continue
    count = ingest_into_mind(mind_name, [text], source="wikipedia", minds_dir=MINDS_DIR)
    print(f"  Indexed {count} entries into '{mind_name}'")


# ── Step 2: Check if Ollama is available ────────────────────────────

def _ollama_available() -> bool:
    """Return True if Ollama responds on localhost:11434."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"User-Agent": "solari-demo"},
        )
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


has_ollama = _ollama_available()
if has_ollama:
    print("\n[dream] Ollama detected — will run full NREM + REM cycle")
else:
    print("\n[dream] Ollama not running — will run NREM only (bridge discovery)")
    print("        To enable hypothesis generation, start Ollama with a model:")
    print("            ollama pull qwen2.5:7b && ollama serve")


# ── Step 3: Run the dream engine ────────────────────────────────────

from solari.dream import DreamEngine

engine = DreamEngine(
    minds_dir=MINDS_DIR,
    output_path=os.path.join(MINDS_DIR, "dream_demo_insights.jsonl"),
)

# NREM: find structural bridges between domains
print("\n" + "─" * 60)
print("  NREM PHASE: Cross-domain consolidation")
print("─" * 60)
bridges = engine.nrem_phase()

if not bridges:
    print("\nNo bridges found. Try adding more diverse topics.")
else:
    print(f"\nDiscovered {len(bridges)} cross-domain bridge(s):\n")
    for i, b in enumerate(bridges, 1):
        print(f"  Bridge {i}: {b['mind_a']} <-> {b['mind_b']}")
        print(f"    Similarity: {b['similarity']:.3f}")
        print(f"    Probe: {b['probe'][:80]}")
        print()

    # REM: synthesize insights from bridges (requires Ollama)
    if has_ollama:
        print("─" * 60)
        print("  REM PHASE: Hypothesis generation")
        print("─" * 60)
        hypotheses = engine.rem_phase(bridges)
        if hypotheses:
            print(f"\nGenerated {len(hypotheses)} insight(s):\n")
            for i, h in enumerate(hypotheses, 1):
                insight = h.get("insight", h.get("hypothesis", "(no text)"))
                print(f"  Insight {i}:")
                print(f"    {insight[:300]}")
                print()
        else:
            print("\nNo novel hypotheses generated this cycle.")

# ── Summary ─────────────────────────────────────────────────────────

print("─" * 60)
state = engine.state
print(f"  Dream complete.")
print(f"    Total bridges found:      {state.get('bridges_found', len(bridges))}")
print(f"    Hypotheses generated:     {state.get('hypotheses_generated', 0)}")
print(f"    Insights file:            {engine.output_path}")
print("─" * 60)
