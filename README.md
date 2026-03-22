<p align="center">
  <img src="logo.png" alt="Solari" width="180"/>
</p>

<h1 align="center">Solari</h1>
<p align="center"><strong>Give your AI a brain, not just hands.</strong></p>
<p align="center">
  <em>Persistent memory, grounded knowledge, and safe automation for any LLM.<br>Works with Claude, GPT, Codex, Ollama, or any provider you already use.</em>
</p>

<p align="center">
  <a href="https://github.com/SolariResearch/solari/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/SolariResearch/solari/stargazers"><img src="https://img.shields.io/github/stars/SolariResearch/solari?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#what-it-does">What It Does</a> &bull;
  <a href="#bring-your-own-ai">Bring Your Own AI</a> &bull;
  <a href="#the-dream-engine">Dream Engine</a> &bull;
  <a href="#safe-by-design">Safe by Design</a> &bull;
  <a href="https://solarisystems.net">Enterprise</a>
</p>

---

## The Problem

Your AI forgets everything between sessions. It hallucinates about your domain. And if you give it the ability to act, it might get your accounts banned.

Solari fixes all three.

## What It Does

Solari is **persistent memory and grounded knowledge** for any AI you already use. It's not a replacement for Claude, GPT, or Codex — it's what makes them actually useful.

```bash
# Install
git clone https://github.com/SolariResearch/solari.git
cd solari && pip install -e .

# Feed it your knowledge
solari ingest --pdf research_paper.pdf --mind physics
solari ingest --url "https://docs.your-project.com" --mind my_project
solari ingest --youtube "https://youtube.com/watch?v=..." --mind lectures

# Ask questions grounded in YOUR knowledge
solari query "explain the key findings" --mind physics
```

Your AI now has expert-level knowledge in whatever you fed it. It persists forever. It never hallucinates on covered domains. And it works with **any** LLM you point it at.

**Try it right now** with the included starter minds (1,767 entries across programming, biology, and physics):

```bash
solari query "how does a hash table handle collisions" --minds-dir starter-minds
solari query "how does the immune system fight infection" --minds-dir starter-minds
solari query "what is entropy in thermodynamics" --minds-dir starter-minds
```

---

## Bring Your Own AI

Solari is not another AI provider. It extends the one you already pay for.

```bash
# Works with your Claude subscription
solari agent --provider anthropic --model claude-sonnet-4-20250514

# Works with OpenAI / Codex
solari agent --provider openai --model gpt-4o

# Works with local models (Ollama)
solari agent --provider ollama --model qwen2.5:7b

# Works with any OpenAI-compatible endpoint
solari agent --provider custom --base-url http://localhost:8080/v1
```

Set your API key once:

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

Solari handles the memory. Your provider handles the intelligence. Together they're unstoppable.

---

## Quick Start

### 1. Install

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

Grounded, sourced answers from the knowledge you ingested. No hallucination.

### 4. Run the Dream Engine

```bash
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

Each "mind" is a local vector index. Stack them. Query across them. They persist forever.

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

**Measured improvement:** A 7B model grounded by Solari scored **60%** on domain questions vs **40%** without. Same model, same questions. The knowledge layer is the difference.

### The Dream Engine

This is what nobody else has.

The Dream Engine takes separate knowledge bases and finds **cross-domain connections** that no single expert would see.

```bash
solari dream --minds physics,economics,biology --cycles 5
```

**How it works:**
1. **NREM phase** — probes pairs of knowledge bases with shared questions, finds hidden structural bridges
2. **REM phase** — feeds bridges into your LLM to generate novel hypotheses
3. **Parliament mode** — expert viewpoints debate, dissent is measured, synthesis emerges

In production testing: **1,400+ genuine cross-domain insights** generated — connections between immunology and cybersecurity, physics and economics, game theory and software architecture.

### Global Workspace

A production implementation of [Global Workspace Theory](https://en.wikipedia.org/wiki/Global_workspace_theory) for building cognitive agents.

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

gw = GlobalWorkspace(capacity=7)
gw.register_processor(ThreatDetector())
result = gw.tick()
```

Attention competition, coherence scoring, narrative threading, meta-cognition, phenomenal state. Used in a system that has run 2,900+ autonomous cycles in production.

---

## Safe by Design

Other agent tools give AI the ability to act first and understand later. That's how people get their Google accounts banned, their credentials stolen, and their inboxes spammed.

Solari is different:

- **Knowledge first, action second** — the AI queries your minds before doing anything, so it actually understands the context
- **You approve actions** — nothing happens without your confirmation
- **No marketplace of unvetted plugins** — no supply chain attacks through malicious skills
- **Your data stays yours** — knowledge is stored locally, not sent to third-party servers
- **No autonomous background processes** — Solari responds when you ask, it doesn't act on its own

This isn't about limiting what AI can do. It's about making sure it **knows what it's doing** before it does it.

---

## Architecture

```
                    ┌──────────────────┐
                    │  solari ingest   │ ← PDFs, URLs, YouTube, arXiv
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │     Minds        │ ← Vector indices, <50ms lookup
                    │  (local storage) │
                    └──┬──────────┬────┘
                       │          │
              ┌────────▼──┐  ┌───▼────────┐
              │solari query│  │solari dream│
              │ (retrieve) │  │(synthesize)│
              └─────┬──────┘  └──────┬─────┘
                    │                │
              ┌─────▼────────────────▼─────┐
              │     Your AI Provider       │
              │ (Claude / GPT / Ollama)    │
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
| Starter minds included | 1,767 entries (programming, biology, physics) |
| Dream insights | 1,400+ generated in production |
| Providers supported | Claude, GPT, Codex, Ollama, any OpenAI-compatible API |

---

## Why Solari?

| Problem | Solari |
|---------|--------|
| AI forgets between sessions | Minds persist forever — knowledge compounds |
| AI hallucinate on your domain | Grounded responses from YOUR verified knowledge |
| Agent tools get accounts banned | Safe by design — you approve every action |
| Locked into one AI provider | Works with any LLM — bring your own API key |
| RAG needs infrastructure | No Docker, no database, no server — just `pip install` |
| Knowledge stays siloed | Dream Engine bridges domains automatically |

---

## Examples

See the [`examples/`](examples/) directory:

- **[quickstart.py](examples/quickstart.py)** — Ingest and query in 30 lines
- **[dream_demo.py](examples/dream_demo.py)** — Cross-domain synthesis in action
- **[workspace_demo.py](examples/workspace_demo.py)** — Build a cognitive architecture

---

## About

Solari is built by [Solari Systems](https://solarisystems.net), extracted from a production autonomous intelligence system with 15 months of R&D behind it.

These are real tools that run in production every day — not prototypes, not demos. Each module works independently. Together they form something greater.

## Support

If Solari saves you time, consider supporting development:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/solarisystems)

For enterprise deployments with managed hosting, team features, and priority support, visit [solarisystems.net](https://solarisystems.net).

## License

AGPL-3.0 — Free for open-source use. [Commercial licensing](https://solarisystems.net) available.

---

<p align="center">
  <strong>Built by <a href="https://solarisystems.net">Solari Systems</a></strong><br>
  <em>Give your AI a brain, not just hands.</em>
</p>
