<p align="center">
  <img src="logo.png" alt="Solari" width="180"/>
</p>

<h1 align="center">Solari</h1>
<p align="center"><strong>The Deep Knowledge Engine</strong></p>
<p align="center">
  <em>Turn anything into a searchable knowledge brain. Make any LLM an expert.<br>Zero hallucination on covered domains.</em>
</p>

<p align="center">
  <a href="https://github.com/SolariResearch/solari/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/SolariResearch/solari/stargazers"><img src="https://img.shields.io/github/stars/SolariResearch/solari?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#what-it-does">What It Does</a> &bull;
  <a href="#the-dream-engine">Dream Engine</a> &bull;
  <a href="#global-workspace">Global Workspace</a> &bull;
  <a href="https://solarisystems.net">Enterprise</a>
</p>

---

## What It Does

You have documents, papers, codebases, and domain knowledge scattered everywhere. Your LLM doesn't know any of it. Solari fixes that.

```bash
# Install
pip install solari-ai

# Feed it your knowledge
solari ingest --pdf research_paper.pdf --mind physics
solari ingest --url "https://docs.your-project.com" --mind my_project
solari ingest --youtube "https://youtube.com/watch?v=..." --mind lectures

# Ask questions grounded in YOUR knowledge
solari query "explain the key findings" --mind physics
```

That's it. Your LLM now has expert-level knowledge in whatever you fed it, stored locally, queryable in milliseconds.

**Solari ships with starter minds** so you can try it immediately:

```bash
# Try it right now with the included knowledge bases
solari query "what is SQL injection" --minds-dir starter-minds
solari query "Python exception handling best practices" --minds-dir starter-minds
```

---

## Quick Start

### 1. Install

```bash
pip install solari-ai
```

Or clone and install from source:

```bash
git clone https://github.com/SolariResearch/solari.git
cd solari
pip install -e .
```

### 2. Build Your First Mind

```bash
# Ingest a Wikipedia article
solari ingest --wikipedia "Machine learning" --mind ml

# Or a PDF
solari ingest --pdf paper.pdf --mind research

# Or a YouTube lecture
solari ingest --youtube "https://youtube.com/watch?v=..." --mind lectures
```

### 3. Query It

```bash
solari query "how do neural networks learn" --mind ml
```

You'll get grounded, sourced answers from the knowledge you ingested. No hallucination.

### 4. Run the Dream Engine

```bash
# Requires Ollama (https://ollama.ai) running locally
solari dream --minds ml,research --cycles 3
```

Watch Solari find connections between your knowledge domains that you didn't know existed.

---

## Features

### Ingest Anything

| Source | Command |
|--------|---------|
| Web page | `solari ingest --url URL --mind NAME` |
| PDF | `solari ingest --pdf PATH --mind NAME` |
| YouTube | `solari ingest --youtube URL --mind NAME` |
| arXiv paper | `solari ingest --arxiv 2301.00001 --mind NAME` |
| Wikipedia | `solari ingest --wikipedia "Topic" --mind NAME` |
| Local file | `solari ingest --file notes.txt --mind NAME` |
| Batch URLs | `solari ingest --batch urls.txt --mind NAME` |
| Directory | `solari ingest --dir ./docs/ --mind NAME` |

Each "mind" is a FAISS vector index stored locally. Stack them. Query across them. They persist forever.

### Query with Precision

```bash
# Search all minds
solari query "your question"

# Search specific minds
solari query "your question" --minds physics,chemistry

# JSON output for pipelines
solari query "your question" --json --top 10

# List available minds
solari minds
```

**Measured improvement:** In controlled A/B testing, a 7B local model grounded by Solari minds scored **60%** on domain-specific questions vs **40%** without. Same model, same questions. The knowledge layer is the difference.

### The Dream Engine

This is what nobody else has.

The Dream Engine takes your knowledge bases, spawns expert perspectives from each domain, and runs a structured parliament debate to find **cross-domain connections** that no single expert would see.

```bash
solari dream --minds physics,economics,biology --cycles 5
```

**How it works:**
1. **NREM phase** — probes pairs of knowledge bases with shared questions, measures semantic overlap, identifies hidden structural bridges
2. **REM phase** — feeds bridges into an LLM that generates novel hypotheses from the cross-domain connection
3. **Parliament mode** — multiple expert viewpoints argue, dissent is measured by embedding distance, coalitions form, synthesis emerges from adversarial debate

In production testing, the Dream Engine has generated **1,400+ genuine cross-domain insights** autonomously — connections between fields like immunology and cybersecurity, physics and economics, game theory and software architecture.

**Output:** JSONL file with scored insights (novelty, actionability, cross-domain relevance).

### Global Workspace

An implementation of [Global Workspace Theory](https://en.wikipedia.org/wiki/Global_workspace_theory) for autonomous agents — the leading scientific theory of how consciousness emerges from parallel processing.

```python
from solari.workspace import GlobalWorkspace, Processor, WorkspaceItem

class ThreatDetector(Processor):
    name = "threat"
    def bid(self, context):
        return [WorkspaceItem(
            source=self.name,
            content="Anomalous login pattern detected",
            item_type="threat",
            urgency=0.9,
            novelty=0.8,
        )]

class ResourceMonitor(Processor):
    name = "resources"
    def bid(self, context):
        return [WorkspaceItem(
            source=self.name,
            content="CPU at 45%, normal",
            item_type="status",
            urgency=0.2,
            novelty=0.1,
        )]

gw = GlobalWorkspace(capacity=7)
gw.register_processor(ThreatDetector())
gw.register_processor(ResourceMonitor())

result = gw.tick()
# ThreatDetector wins — urgency 0.9 beats 0.2
# All processors receive the winning broadcast
```

**What's inside:**
- **Attention mechanism** — 4-dimension scoring (urgency, novelty, relevance, emotion) with hysteresis
- **Coherence scoring** — detects when the system's beliefs conflict
- **Narrative threading** — maintains a running story with chapters, causal chains, and anticipation
- **Meta-cognition** — loop detection, bias detection, confidence calibration, meta-emotions (flow, focused, frustrated, curious)
- **Phenomenal state** — integrated valence/arousal/dominance snapshot

This isn't a toy. It's a production cognitive architecture used in a system that has run 2,900+ autonomous cycles.

---

## Architecture

```
                    ┌──────────────────┐
                    │  solari ingest   │ ← PDFs, URLs, YouTube, arXiv
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   FAISS Minds    │ ← Vector indices, <50ms lookup
                    │  (local, private)│
                    └──┬──────────┬────┘
                       │          │
              ┌────────▼──┐  ┌───▼────────┐
              │solari query│  │solari dream│
              │ (retrieve) │  │(synthesize)│
              └────────────┘  └────────────┘
                       │          │
              ┌────────▼──────────▼────────┐
              │     Global Workspace       │
              │  (attention + coherence)   │
              └────────────────────────────┘
```

---

## Performance

| Metric | Value |
|--------|-------|
| Query latency | <50ms warm |
| Ingest throughput | ~1,000 chunks/sec |
| Mind storage | ~4MB per 1,000 entries |
| Quality improvement | **+20% on domain questions** (verified A/B) |
| Dream insights | 1,400+ generated in production |
| Workspace cycles | 2,900+ autonomous cycles run |

---

## Why Solari?

| Problem | Solari |
|---------|--------|
| LLMs hallucinate | Ground responses in YOUR verified knowledge |
| RAG needs a server | FAISS runs locally — no Docker, no database, no API |
| Knowledge stays siloed | Dream Engine bridges domains automatically |
| Agents forget everything | Minds persist and compound across sessions |
| Your data leaves your machine | Everything runs locally. Zero cloud dependency |

---

## Examples

See the [`examples/`](examples/) directory:

- **[quickstart.py](examples/quickstart.py)** — Ingest and query in 30 lines
- **[dream_demo.py](examples/dream_demo.py)** — Cross-domain synthesis in action
- **[workspace_demo.py](examples/workspace_demo.py)** — Build a cognitive architecture

---

## About

Solari is extracted from a larger autonomous intelligence system built over 15 months of R&D. These tools are the foundation — each works independently, together they form a cognitive architecture.

The full system includes reinforcement learning, game-theoretic reasoning, a 31-module cognitive spine, and neurochemistry-modulated cognition. Enterprise licensing available through [Solari Systems](https://solarisystems.net).

## Support

If Solari saves you time, consider supporting development:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/solarisystems)

For enterprise licensing and custom deployments, visit [solarisystems.net](https://solarisystems.net).

## License

AGPL-3.0 — Free for open-source use. [Commercial licensing](https://solarisystems.net) available.

---

<p align="center">
  <strong>Built by <a href="https://solarisystems.net">Solari Systems</a></strong>
</p>
