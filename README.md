<p align="center">
  <img src="logo.png" alt="Solari" width="200"/>
</p>

<h1 align="center">SOLARI</h1>
<h3 align="center">The Deep Knowledge Engine</h3>

> Turn anything into a searchable knowledge brain. Make any LLM an expert. Zero hallucination on covered domains.

Solari is a cognitive toolkit for building, querying, and synthesizing local knowledge bases. Feed it PDFs, URLs, YouTube videos, or research papers. It builds FAISS vector indices that any local LLM can query for grounded, hallucination-free answers.

**What makes it different:** Solari doesn't just retrieve — it *thinks*. The Dream Engine finds connections across knowledge domains that no single expert would see. The Global Workspace gives your agent a real cognitive architecture with attention, coherence, and narrative threading.

---

## Quick Start

```bash
pip install solari-ai

# Ingest a research paper
solari ingest --pdf quantum_computing.pdf --mind physics

# Ingest a YouTube lecture
solari ingest --youtube "https://youtube.com/watch?v=..." --mind machine_learning

# Ingest an entire documentation site
solari ingest --url "https://docs.example.com" --mind my_project

# Ask questions — grounded in YOUR knowledge, not training data
solari query "explain quantum entanglement" --mind physics

# Find cross-domain connections nobody else would see
solari dream --minds physics,biology,economics --cycles 5
```

---

## The Tools

### `solari ingest` — Feed Solari

Turn any content source into a searchable FAISS knowledge index:

| Source | Command | What It Does |
|--------|---------|--------------|
| PDF | `solari ingest --pdf paper.pdf --mind NAME` | Extracts text, chunks, embeds, indexes |
| URL | `solari ingest --url https://... --mind NAME` | Scrapes content, cleans HTML, indexes |
| YouTube | `solari ingest --youtube URL --mind NAME` | Transcribes audio, indexes transcript |
| arXiv | `solari ingest --arxiv 2301.00001 --mind NAME` | Fetches paper, extracts, indexes |
| Directory | `solari ingest --dir ./docs/ --mind NAME` | Recursively ingests all files |

Minds are stored locally in `./minds/` by default. Each mind is a FAISS index + metadata. Stack them — 10 minds, 100 minds, 1000 minds. They all query in milliseconds.

### `solari query` — Ask Solari

Retrieve grounded knowledge from your minds:

```bash
# Query all minds
solari query "how does TCP handle congestion?"

# Query specific minds
solari query "what causes reentrancy vulnerabilities?" --minds security,solidity

# Get JSON output for pipelines
solari query "market size for AI consulting" --json --top 10
```

**Proven quality improvement:** In controlled A/B testing, a 7B local model with Solari minds scored 60% on domain-specific questions vs 40% for the same model without minds. The knowledge layer makes any model perform above its weight class.

### `solari dream` — Let Solari Think

The Dream Engine is what nobody else has. It takes multiple knowledge domains and algorithmically finds structural bridges between them:

```bash
# Find connections between physics and economics
solari dream --minds physics,economics --cycles 3

# Full parliament debate across all your minds
solari dream --all --cycles 10 --output insights.jsonl
```

**How it works:**
1. Selects knowledge from different domains
2. Spawns expert perspectives from each domain
3. Runs a parliament debate — advocates argue, skeptics challenge
4. Synthesizes insights that survive adversarial scrutiny
5. Scores by novelty, actionability, and cross-domain relevance

In production, the Dream Engine has generated 1,400+ genuine cross-domain insights autonomously.

### `solari workspace` — Give Your Agent a Brain

An implementation of Global Workspace Theory for autonomous agents:

```python
from solari import GlobalWorkspace, Processor

# Create processors that compete for attention
class SecurityScanner(Processor):
    def process(self, broadcast):
        # Analyze for security issues
        return {"urgency": 0.9, "finding": "SQL injection in auth.py"}

class PerformanceMonitor(Processor):
    def process(self, broadcast):
        return {"urgency": 0.3, "status": "normal"}

# Workspace handles attention competition
ws = GlobalWorkspace(processors=[SecurityScanner(), PerformanceMonitor()])
ws.tick()  # SecurityScanner wins — higher urgency
# All processors receive the broadcast
```

Features:
- **Attention mechanism** — processors bid based on urgency, novelty, relevance, emotion
- **Coherence scoring** — detects when the system's beliefs conflict
- **Narrative threading** — maintains a running story with causal chain
- **Meta-cognition** — loop detection, bias detection, meta-emotions

---

## Architecture

```
                    ┌─────────────────┐
                    │   solari ingest    │  ← PDFs, URLs, YouTube, arXiv
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   FAISS Minds   │  ← Vector indices, millisecond lookup
                    │  (local, private)│
                    └──┬─────────┬────┘
                       │         │
              ┌────────▼──┐  ┌──▼────────┐
              │ solari query │  │ solari dream │
              │ (retrieve)│  │(synthesize)│
              └───────────┘  └───────────┘
                       │         │
              ┌────────▼─────────▼────────┐
              │    Global Workspace       │
              │  (attention + coherence)  │
              └───────────────────────────┘
```

---

## Why Solari?

| Problem | How Solari Solves It |
|---------|----------------------|
| LLMs hallucinate | Ground responses in YOUR verified knowledge |
| RAG is slow and complex | FAISS indices query in <50ms, no server needed |
| Knowledge silos | Dream Engine bridges domains automatically |
| Agents have no memory | Minds persist across sessions, compound over time |
| Cloud dependency | Everything runs locally. Your data never leaves your machine |

---

## Requirements

- Python 3.10+
- `pip install solari-ai`
- Optional: [Ollama](https://ollama.ai) for local LLM synthesis in Dream Engine

---

## Performance

| Metric | Value |
|--------|-------|
| Query latency | <50ms (warm FAISS) |
| Ingest speed | ~1000 chunks/second |
| Mind size | ~4MB per 1000 entries |
| Quality improvement | +20% on domain questions (verified A/B) |

---

## From the Creators

Solari is a standalone toolkit extracted from a larger autonomous intelligence system built over 15 months. Each tool works independently. Together they form a cognitive architecture.

The full system — with reinforcement learning, game-theoretic reasoning, and a 31-module cognitive spine — is available for enterprise licensing through [Solari Systems](https://solarisystems.net).

---

## License

AGPL-3.0 — Free for open-source use. Commercial licensing available.

---

*Built by [Solari Systems](https://solarisystems.net)*
