#!/usr/bin/env python3
"""
Cross-Domain Knowledge Synthesis Engine ("Dream Engine")

A biologically-inspired system that finds hidden connections between
separate knowledge bases (FAISS vector indices) by having "expert"
perspectives from different domains debate each other.

Architecture (inspired by Global Workspace Theory and neuroscience):

  Phase 1 -- NREM (Consolidation):
    Scan knowledge bases, probe pairs with shared queries, measure
    semantic similarity of results. Pairs with high similarity form
    "cross-domain bridges" -- evidence that two unrelated domains
    share hidden structure.

  Phase 2 -- REM (Hypothesis Generation):
    Feed bridges into an LLM with a parliament/debate prompt. Multiple
    expert viewpoints (drawn from different knowledge domains) argue
    and synthesize. The LLM generates novel hypotheses that emerge
    from the cross-domain connection.

  Parliament Mode (full debate):
    Multiple knowledge bases are queried in parallel. Their responses
    are fed into a structured debate: expert statements, dissent
    detection (which experts disagree most), coalition detection
    (which experts cluster), multi-round rebuttals, and final
    synthesis. Disagreement between experts is treated as the
    highest-value signal.

Key concepts:
  - Minds: FAISS indices with associated metadata (JSON). Each
    represents a domain of knowledge.
  - Lenses: Abstract questions about systems (e.g. "how failures
    cascade") used to probe minds from a specific angle.
  - Bridges: Discovered connections between two minds that respond
    similarly to the same probe.
  - Parliament: Structured debate where expert responses are
    aggregated, dissent is measured via embedding distance, and
    an LLM synthesizes the cross-domain insight.

Requirements:
  pip install sentence-transformers faiss-cpu numpy requests

Usage:
  # Run 3 dream cycles over minds in ./minds/
  python dream.py --minds-dir ./minds/ --cycles 3

  # Use a specific Ollama model
  python dream.py --minds-dir ./minds/ --model qwen3:8b

  # NREM only (find bridges, no LLM synthesis)
  python dream.py --minds-dir ./minds/ --nrem-only

  # REM only (synthesize from cached bridges)
  python dream.py --minds-dir ./minds/ --rem-only

  # Show current state
  python dream.py --minds-dir ./minds/ --status

  # Parliament mode with full multi-round debate
  python dream.py --minds-dir ./minds/ --parliament --cycles 1

  # Custom output file
  python dream.py --minds-dir ./minds/ --output insights.jsonl
"""

import argparse
import concurrent.futures
import gzip
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.7
MAX_BRIDGES_PER_CYCLE = 20
MAX_HYPOTHESES_PER_CYCLE = 10
NOVELTY_THRESHOLD = 0.75  # insights more similar than this are "stale"

# Cross-domain probe query templates (rotated each cycle)
PROBE_TEMPLATES = [
    "What patterns connect {mind_a} and {mind_b}?",
    "How does {concept_from_a} apply to {domain_of_b}?",
    "What failure modes in {mind_a} are analogous to {mind_b}?",
    "What optimization technique from {mind_a} could improve {mind_b}?",
]

# Abstract lenses -- universal questions about systems used to probe minds
ABSTRACT_LENSES = [
    "how failures cascade and propagate through connected systems",
    "optimization under tight constraints with competing priorities",
    "pattern recognition and detecting anomalies in noisy data",
    "feedback loops that amplify or dampen system behavior",
    "resource allocation when multiple agents compete for the same pool",
    "trust and verification in environments where actors may be adversarial",
    "emergent behavior arising from simple local rules",
    "compression tradeoffs -- what information do you lose and what do you keep",
    "timing and ordering -- when sequence matters more than content",
    "layered defense where no single layer is sufficient alone",
    "adaptation under pressure -- how systems evolve when stressed",
    "coordination without central authority or shared state",
    "error recovery -- how systems detect and correct their own mistakes",
    "abstraction boundaries -- where simplified models break down",
    "incentive alignment -- making self-interest serve collective goals",
    "state management -- tracking what is true right now across distributed actors",
    "boundary conditions -- where normal behavior transitions to failure",
    "information asymmetry -- when one party knows more than another",
    "composability -- building complex behavior from simple primitives",
    "invariant preservation -- what must always be true regardless of inputs",
]

# Relevance tags -- insights touching these get flagged
RELEVANCE_TAGS = frozenset({
    "security", "optimization", "architecture", "engineering",
    "vulnerability", "design", "testing", "performance",
    "reliability", "scalability", "data", "analysis",
})

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("dream")


# ---------------------------------------------------------------------------
# Embedding & FAISS Engine (in-memory, singleton)
# ---------------------------------------------------------------------------

class CognitionEngine:
    """
    Persistent in-memory engine. Loads encoder + FAISS indices once and
    keeps them hot for fast repeated queries (~0.15s per query instead
    of seconds for cold subprocess calls).
    """

    _instance = None

    @classmethod
    def get(cls, minds_dir: str = "."):
        if cls._instance is None:
            cls._instance = cls(minds_dir)
        return cls._instance

    def __init__(self, minds_dir: str):
        import faiss  # noqa: F811 (lazy import is intentional)
        self._faiss = faiss
        self._minds_dir = minds_dir
        self._encoder = None
        self._np = None
        self._indexes: Dict[str, tuple] = {}  # mind -> (faiss_index, metadata)

    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._encoder = SentenceTransformer(DEFAULT_EMBED_MODEL)
            self._np = np
        return self._encoder

    def _load_index(self, mind: str):
        """Load a FAISS index + metadata into memory. Cached."""
        if mind in self._indexes:
            return self._indexes[mind]
        idx_path = os.path.join(self._minds_dir, mind, "index.faiss")
        meta_gz = os.path.join(self._minds_dir, mind, "metadata.json.gz")
        meta_json = os.path.join(self._minds_dir, mind, "metadata.json")
        if not os.path.exists(idx_path):
            return None, None
        try:
            idx = self._faiss.read_index(idx_path)
            meta: list = []
            if os.path.exists(meta_gz):
                with gzip.open(meta_gz, "rt") as f:
                    meta = json.load(f)
            elif os.path.exists(meta_json):
                with open(meta_json) as f:
                    meta = json.load(f)
            # Cap cache at 30 indexes to manage memory
            if len(self._indexes) >= 30:
                oldest = next(iter(self._indexes))
                del self._indexes[oldest]
            self._indexes[mind] = (idx, meta)
            return idx, meta
        except Exception:
            return None, None

    def query(self, query_text: str, mind: str, top: int = 3) -> str:
        """Direct in-memory FAISS query. Returns formatted result text."""
        encoder = self._ensure_encoder()
        np = self._np
        idx, meta = self._load_index(mind)
        if idx is None or not meta:
            return ""
        try:
            emb = encoder.encode([query_text], normalize_embeddings=True).astype("float32")
            D, I = idx.search(emb, min(top, idx.ntotal))
            results = []
            for rank, (dist, idx_val) in enumerate(zip(D[0], I[0])):
                if idx_val < 0 or idx_val >= len(meta):
                    continue
                entry = meta[idx_val]
                text = entry.get("text", entry.get("content", ""))
                sim = float(dist)
                results.append(
                    f"--- Result {rank + 1} [{mind}] (sim={sim:.3f}) ---\n{text[:500]}"
                )
            return "\n\n".join(results)
        except Exception as e:
            return f"[error: {e}]"

    def get_mind_names(self) -> List[str]:
        """List available minds (directories with index.faiss)."""
        try:
            return [
                d
                for d in os.listdir(self._minds_dir)
                if os.path.isdir(os.path.join(self._minds_dir, d))
                and os.path.exists(os.path.join(self._minds_dir, d, "index.faiss"))
            ]
        except Exception:
            return []

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two text strings."""
        encoder = self._ensure_encoder()
        np = self._np
        embeddings = encoder.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))

    def embed(self, text: str):
        """Get embedding vector for text."""
        encoder = self._ensure_encoder()
        return encoder.encode([text], normalize_embeddings=True).astype("float32")[0]

    def invalidate_cache(self, mind: str):
        """Drop cached index so next query reloads from disk."""
        self._indexes.pop(mind, None)


# ---------------------------------------------------------------------------
# LLM Interface (Ollama)
# ---------------------------------------------------------------------------

def _llm_call(
    prompt: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    timeout: int = 120,
    system: str = "You are a cross-domain knowledge synthesis engine.",
) -> Optional[str]:
    """Call Ollama's generate endpoint. Returns response text or None."""
    try:
        import requests

        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Expert Population -- parallel mind queries
# ---------------------------------------------------------------------------

class ExpertPopulation:
    """
    Queries multiple knowledge bases (minds) in parallel, like Global
    Workspace Theory's audience: many specialized processors sitting
    in the dark, all processing the spotlight content at once.
    """

    def __init__(self, engine: CognitionEngine):
        self.engine = engine

    def consult_experts(
        self, query: str, mind_names: List[str], top_per_mind: int = 2
    ) -> Dict[str, str]:
        """Query many minds in parallel. Returns dict of mind -> response."""
        results: Dict[str, str] = {}

        def query_one(mind):
            try:
                return mind, self.engine.query(query, mind, top=top_per_mind)
            except Exception:
                return mind, ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(query_one, m) for m in mind_names[:20]]
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    mind, response = future.result(timeout=10)
                    if response and len(response) > 30 and "[error" not in response:
                        results[mind] = response
                except Exception:
                    pass
        return results

    def find_dissent(
        self, expert_responses: Dict[str, str], query: str
    ) -> List[Tuple[str, str, float]]:
        """
        Find disagreement between experts -- the most valuable signal.
        Experts that give very different answers to the same query
        indicate unexplored territory between domains.
        Returns list of (mind_a, mind_b, similarity) sorted by lowest
        similarity first (maximum dissent).
        """
        if len(expert_responses) < 3:
            return []

        import numpy as np

        encoder = self.engine._ensure_encoder()
        texts = []
        minds = []
        for mind, response in expert_responses.items():
            texts.append(response[:300])
            minds.append(mind)

        if len(texts) < 3:
            return []

        embeddings = encoder.encode(texts, normalize_embeddings=True)
        sims = np.dot(embeddings, embeddings.T)

        dissent_pairs = []
        for i in range(len(minds)):
            for j in range(i + 1, len(minds)):
                dissent_pairs.append((minds[i], minds[j], float(sims[i][j])))

        dissent_pairs.sort(key=lambda x: x[2])  # lowest similarity first
        return dissent_pairs[:3]


# ---------------------------------------------------------------------------
# Parliament -- debate, consensus, and synthesis
# ---------------------------------------------------------------------------

class Parliament:
    """
    The Global Workspace as parliament: experts broadcast responses,
    the parliament measures consensus via embedding similarity, detects
    dissent, runs multi-round debate, and produces a final synthesis.

    Disagreement between experts is the highest-value signal.
    """

    def deliberate(
        self,
        query: str,
        expert_responses: Dict[str, str],
        dissent_pairs: List[Tuple[str, str, float]],
        engine: CognitionEngine,
    ) -> dict:
        """
        Run a parliamentary deliberation with real consensus measurement
        using embedding similarity across expert responses.
        """
        if not expert_responses:
            return {"consensus_strength": 0, "experts_consulted": 0}

        n_experts = len(expert_responses)
        n_responding = sum(1 for v in expert_responses.values() if len(v) > 50)

        real_consensus = 0.5
        avg_similarity = 0.0
        if n_responding >= 3:
            try:
                import numpy as np

                encoder = engine._ensure_encoder()
                texts = [r[:300] for r in expert_responses.values() if len(r) > 50]
                if len(texts) >= 3:
                    embeddings = encoder.encode(texts, normalize_embeddings=True)
                    sims = np.dot(embeddings, embeddings.T)
                    n = len(texts)
                    total = 0.0
                    count = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            total += float(sims[i][j])
                            count += 1
                    avg_similarity = total / max(1, count)
                    real_consensus = max(0, min(1, (avg_similarity + 0.1) / 0.7))
            except Exception:
                pass

        return {
            "query": query[:100],
            "experts_consulted": n_experts,
            "experts_responding": n_responding,
            "consensus_strength": round(real_consensus, 3),
            "avg_similarity": round(avg_similarity, 4),
            "dissent_pairs": [
                (a, b, round(s, 3)) for a, b, s in dissent_pairs[:3]
            ],
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def synthesize_debate(
        self,
        expert_responses: Dict[str, str],
        dissent_pairs: List[Tuple[str, str, float]],
        lens: str,
    ) -> str:
        """
        Build a structured debate text from expert responses.
        Highlights dissent (most valuable signal) and formats
        for LLM synthesis.
        """
        parts = []
        for mind, response in list(expert_responses.items())[:8]:
            parts.append(f"[{mind}]: {response[:250]}")

        debate_section = ""
        if dissent_pairs:
            debate_section = (
                "\n\nDISSENT (experts that DISAGREE -- highest-value signal):\n"
            )
            for a, b, sim in dissent_pairs[:3]:
                debate_section += f"  {a} vs {b} (similarity: {sim:.3f})\n"
                debate_section += (
                    f"  [{a}]: {expert_responses.get(a, '')[:200]}\n"
                )
                debate_section += (
                    f"  [{b}]: {expert_responses.get(b, '')[:200]}\n\n"
                )

        expert_text = "\n\n".join(parts)
        return f"EXPERT RESPONSES:\n{expert_text}{debate_section}"

    def multi_round_debate(
        self,
        initial_synthesis: str,
        dissenting_minds: List[str],
        expert_responses: Dict[str, str],
        engine: CognitionEngine,
        ollama_url: str,
        model: str,
        rounds: int = 2,
    ) -> str:
        """
        Multi-round debate: feed initial synthesis back to dissenting
        experts for refinement. Each round sharpens the insight.

        Round 1 = opening statements, Round 2 = rebuttals,
        Round 3 = final position.
        """
        if not dissenting_minds or not initial_synthesis:
            return initial_synthesis

        current_synthesis = initial_synthesis

        for round_num in range(rounds):
            rebuttals: Dict[str, str] = {}
            for mind in dissenting_minds[:4]:
                try:
                    rebuttal_query = (
                        f"Challenge this claim: {current_synthesis[:200]}"
                    )
                    response = engine.query(rebuttal_query, mind, top=2)
                    if response and len(response) > 50:
                        rebuttals[mind] = response
                except Exception:
                    pass

            if not rebuttals:
                break

            rebuttal_text = "\n".join(
                f"[{m} REBUTTAL]: {r[:200]}" for m, r in rebuttals.items()
            )

            debate_prompt = f"""Round {round_num + 2} of expert debate.

Previous synthesis: {current_synthesis[:400]}

Expert rebuttals:
{rebuttal_text}

Integrate the rebuttals. Where do they STRENGTHEN the original claim?
Where do they WEAKEN it? What new angle do they reveal?

REFINED_SYNTHESIS: [updated claim integrating rebuttals]
STRENGTHENED: [aspects confirmed by rebuttals]
WEAKENED: [aspects challenged by rebuttals]
NEW_ANGLE: [novel perspective from the debate]"""

            refined = _llm_call(
                debate_prompt,
                ollama_url=ollama_url,
                model=model,
                temperature=0.7,
                max_tokens=800,
                timeout=60,
            )
            if refined and "[error" not in refined and len(refined) > 100:
                current_synthesis = refined
            else:
                break

        return current_synthesis


# ---------------------------------------------------------------------------
# Coalition Detector
# ---------------------------------------------------------------------------

class CoalitionDetector:
    """
    Detects coalitions (factions) among experts: groups of minds that
    give similar responses. Like political parties -- agreement within
    is expected, disagreement between is where real debates happen.
    """

    def detect_coalitions(
        self, expert_responses: Dict[str, str], engine: CognitionEngine
    ) -> List[List[str]]:
        """
        Cluster experts by response similarity. Returns list of
        coalitions (each a list of mind names).
        """
        if len(expert_responses) < 4:
            return []

        try:
            import numpy as np

            encoder = engine._ensure_encoder()
            minds = list(expert_responses.keys())
            texts = [expert_responses[m][:300] for m in minds]
            embeddings = encoder.encode(texts, normalize_embeddings=True)
            sims = np.dot(embeddings, embeddings.T)

            coalitions: List[List[str]] = []
            assigned: set = set()
            for i in range(len(minds)):
                if minds[i] in assigned:
                    continue
                coalition = [minds[i]]
                assigned.add(minds[i])
                for j in range(i + 1, len(minds)):
                    if minds[j] in assigned:
                        continue
                    if float(sims[i][j]) > 0.3:
                        coalition.append(minds[j])
                        assigned.add(minds[j])
                if len(coalition) >= 2:
                    coalitions.append(coalition)

            return coalitions
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Novelty Detector
# ---------------------------------------------------------------------------

class NoveltyDetector:
    """
    Compares new insights against recent ones using embedding similarity.
    Only genuinely new insights get through -- like the brain's
    habituation response to repeated stimuli.
    """

    def __init__(self):
        self.recent: List[dict] = []
        self.max_recent = 50

    def check_novelty(
        self, synthesis_text: str, engine: CognitionEngine
    ) -> Tuple[bool, float, str]:
        """
        Check if synthesis is genuinely novel compared to recent insights.
        Returns (is_novel, similarity_to_nearest, nearest_text).
        """
        if not self.recent:
            return True, 0.0, ""

        try:
            import numpy as np

            encoder = engine._ensure_encoder()
            new_emb = encoder.encode(
                [synthesis_text[:500]], normalize_embeddings=True
            )
            recent_embs = np.array([r["embedding"] for r in self.recent])
            sims = np.dot(new_emb, recent_embs.T)[0]
            max_idx = int(np.argmax(sims))
            max_sim = float(sims[max_idx])
            nearest_text = self.recent[max_idx].get("text", "")[:100]
            is_novel = max_sim < NOVELTY_THRESHOLD
            return is_novel, max_sim, nearest_text
        except Exception:
            return True, 0.0, ""

    def record_insight(self, synthesis_text: str, engine: CognitionEngine):
        """Record a new insight embedding for future novelty checks."""
        try:
            encoder = engine._ensure_encoder()
            emb = encoder.encode(
                [synthesis_text[:500]], normalize_embeddings=True
            )
            self.recent.append(
                {
                    "text": synthesis_text[:200],
                    "embedding": emb[0].tolist(),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
            if len(self.recent) > self.max_recent:
                self.recent = self.recent[-self.max_recent :]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Schema Evolver -- Hebbian learning on mind-pair connections
# ---------------------------------------------------------------------------

class SchemaEvolver:
    """
    Dynamic schema restructuring: connections that produce insights
    strengthen (Hebbian). Unused connections decay (anti-Hebbian).
    Mirrors neocortical consolidation during sleep.
    """

    def __init__(self):
        self.weights: Dict[str, float] = {}

    def _edge_key(self, a: str, b: str) -> str:
        return f"{min(a, b)}|{max(a, b)}"

    def strengthen(self, mind_a: str, mind_b: str, amount: float = 1.0):
        key = self._edge_key(mind_a, mind_b)
        self.weights[key] = self.weights.get(key, 0) + amount

    def weaken(self, mind_a: str, mind_b: str, amount: float = 0.3):
        key = self._edge_key(mind_a, mind_b)
        current = self.weights.get(key, 0)
        if current > 0:
            self.weights[key] = max(0, current - amount)
            if self.weights[key] == 0:
                del self.weights[key]

    def get_weight(self, mind_a: str, mind_b: str) -> float:
        return self.weights.get(self._edge_key(mind_a, mind_b), 0)

    def strongest_connections(self, top_n: int = 10) -> List[Tuple[str, float]]:
        sorted_edges = sorted(self.weights.items(), key=lambda x: -x[1])
        return sorted_edges[:top_n]

    def suggest_pairs(
        self, minds: List[str], n: int = 5
    ) -> List[Tuple[str, str]]:
        """Suggest mind pairs biased by schema weights."""
        scored = []
        for i, a in enumerate(minds):
            for b in minds[i + 1 :]:
                w = self.get_weight(a, b)
                scored.append((a, b, w))
        scored.sort(key=lambda x: -x[2])
        n_weighted = max(1, int(n * 0.6))
        n_random = n - n_weighted
        weighted = [(a, b) for a, b, w in scored[:n_weighted] if w > 0]
        remaining = [(a, b) for a, b, w in scored if w == 0]
        if remaining:
            random.shuffle(remaining)
            weighted.extend(remaining[:n_random])
        return weighted[:n]


# ---------------------------------------------------------------------------
# Expert Reputation System
# ---------------------------------------------------------------------------

class ExpertReputation:
    """
    Meritocratic reputation: minds that consistently contribute to
    actionable insights gain reputation. High-rep experts get more
    influence in parliament voting.
    """

    def __init__(self):
        self.reputation: Dict[str, float] = {}   # mind -> score 0-100
        self.participation: Dict[str, int] = {}   # mind -> count
        self.successes: Dict[str, int] = {}       # mind -> count

    def record_participation(
        self, minds: List[str], verdict: str
    ):
        for mind in minds:
            self.participation[mind] = self.participation.get(mind, 0) + 1
            if verdict in ("ACTIONABLE", "INTERESTING"):
                self.successes[mind] = self.successes.get(mind, 0) + 1
            total = self.participation[mind]
            succ = self.successes.get(mind, 0)
            score = (succ / max(1, total)) * 100
            # Bayesian prior: decay toward 50
            prior_weight = 5
            score = (score * total + 50 * prior_weight) / (total + prior_weight)
            self.reputation[mind] = round(score, 1)

    def get_reputation(self, mind: str) -> float:
        return self.reputation.get(mind, 50.0)

    def rank_minds(self, mind_list: List[str]) -> List[str]:
        return sorted(mind_list, key=lambda m: -self.get_reputation(m))


# ---------------------------------------------------------------------------
# FAISS Ingestion (write bridges/hypotheses back to a mind)
# ---------------------------------------------------------------------------

def ingest_to_mind(
    mind_name: str,
    entries: List[Dict],
    minds_dir: str,
    engine: CognitionEngine,
) -> int:
    """Ingest entries into a FAISS mind. Returns count ingested."""
    if not entries:
        return 0

    mind_dir = Path(minds_dir) / mind_name
    mind_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata
    meta_path_gz = mind_dir / "metadata.json.gz"
    meta_path = mind_dir / "metadata.json"
    existing: list = []

    if meta_path_gz.exists():
        try:
            with gzip.open(meta_path_gz, "rt") as f:
                existing = json.load(f)
        except Exception:
            pass
    elif meta_path.exists():
        try:
            with open(meta_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    # Deduplicate by content prefix
    existing_texts = {
        e.get("text", e.get("content", ""))[:200] for e in existing
    }
    new_entries = []
    for entry in entries:
        text = entry.get("text", entry.get("content", ""))
        if text[:200] not in existing_texts:
            new_entries.append(entry)
            existing_texts.add(text[:200])

    if not new_entries:
        return 0

    all_entries = existing + new_entries

    try:
        import numpy as np

        encoder = engine._ensure_encoder()
        faiss = engine._faiss

        texts = [e.get("text", e.get("content", "")) for e in all_entries]
        embeddings = encoder.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(mind_dir / "index.faiss"))

        # Write metadata (gzipped)
        with gzip.open(meta_path_gz, "wt") as f:
            json.dump(all_entries, f)

        # Write manifest
        manifest = {
            "mind_id": mind_name,
            "entry_count": len(all_entries),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "domain_tags": ["cross_domain", "synthesis"],
            "source": "dream_engine",
        }
        with open(mind_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        engine.invalidate_cache(mind_name)

        log.info(
            f"Ingested {len(new_entries)} entries into '{mind_name}' "
            f"(total: {len(all_entries)})"
        )
        return len(new_entries)

    except Exception as e:
        log.error(f"FAISS rebuild failed for '{mind_name}': {e}")
        return 0


# ---------------------------------------------------------------------------
# Dream Engine -- main orchestrator
# ---------------------------------------------------------------------------

class DreamEngine:
    """
    Cross-domain knowledge synthesis engine.

    Combines NREM consolidation (finding structural bridges between
    knowledge bases) with REM hypothesis generation (LLM-powered
    synthesis from expert debate).
    """

    def __init__(
        self,
        minds_dir: str,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        output_path: Optional[str] = None,
        state_path: Optional[str] = None,
    ):
        self.minds_dir = os.path.abspath(minds_dir)
        self.ollama_url = ollama_url
        self.model = model
        self.output_path = output_path or os.path.join(
            self.minds_dir, "dream_insights.jsonl"
        )

        # State file lives next to the minds directory
        if state_path:
            self._state_file = state_path
        else:
            self._state_file = os.path.join(
                self.minds_dir, "dream_state.json"
            )

        self.engine = CognitionEngine.get(self.minds_dir)
        self.experts = ExpertPopulation(self.engine)
        self.parliament = Parliament()
        self.coalition_detector = CoalitionDetector()
        self.novelty = NoveltyDetector()
        self.schema = SchemaEvolver()
        self.reputation = ExpertReputation()
        self.state = self._load_state()

    # -- State persistence --------------------------------------------------

    def _load_state(self) -> dict:
        default = {
            "total_dreams": 0,
            "bridges_found": 0,
            "hypotheses_generated": 0,
            "insights_found": 0,
            "last_dream": None,
            "last_bridges": [],
            "probe_index": 0,
            "used_lenses": [],
        }
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file) as f:
                    loaded = json.load(f)
                for k, v in default.items():
                    if k not in loaded:
                        loaded[k] = v
                return loaded
            except Exception:
                pass
        return default

    def _save_state(self):
        os.makedirs(os.path.dirname(self._state_file) or ".", exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _write_insight(self, entry: dict):
        """Append an insight to the output JSONL file."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.output_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # -- Mind discovery ------------------------------------------------------

    def _get_recent_minds(self, hours: int = 24) -> List[Dict]:
        """Find the 10 most recently modified minds."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        mind_times = []

        minds_path = Path(self.minds_dir)
        for mind_dir in minds_path.iterdir():
            if not mind_dir.is_dir() or mind_dir.name.startswith("_"):
                continue
            manifest = mind_dir / "manifest.json"
            if not manifest.exists():
                # Accept minds without manifest if they have index.faiss
                idx = mind_dir / "index.faiss"
                if not idx.exists():
                    continue
                ts = datetime.fromtimestamp(
                    idx.stat().st_mtime, tz=timezone.utc
                )
                mind_times.append(
                    {
                        "name": mind_dir.name,
                        "path": str(mind_dir),
                        "entries": 0,
                        "tags": [],
                        "modified": ts.isoformat(),
                        "modified_dt": ts,
                    }
                )
                continue

            try:
                m = json.load(open(manifest))
                last_updated = m.get("last_updated") or m.get("updated_at")
                ts = None
                if last_updated:
                    try:
                        ts = datetime.fromisoformat(
                            last_updated.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        ts = None
                if ts is None:
                    mtime = manifest.stat().st_mtime
                    ts = datetime.fromtimestamp(mtime, tz=timezone.utc)

                mind_times.append(
                    {
                        "name": m.get("mind_id", mind_dir.name),
                        "path": str(mind_dir),
                        "entries": m.get("entry_count", 0),
                        "tags": m.get("domain_tags", []),
                        "modified": ts.isoformat(),
                        "modified_dt": ts,
                    }
                )
            except Exception:
                continue

        mind_times.sort(key=lambda x: x["modified_dt"], reverse=True)

        recent = [m for m in mind_times if m["modified_dt"] >= cutoff]
        if len(recent) < 3:
            recent = mind_times[:10]
        else:
            recent = recent[:10]

        for m in recent:
            del m["modified_dt"]

        return recent

    def _get_probe_query(self, mind_a: str, mind_b: str) -> str:
        idx = self.state["probe_index"] % len(PROBE_TEMPLATES)
        template = PROBE_TEMPLATES[idx]
        return template.format(
            mind_a=mind_a.replace("_", " "),
            mind_b=mind_b.replace("_", " "),
            concept_from_a=mind_a.replace("_", " "),
            domain_of_b=mind_b.replace("_", " "),
        )

    def _pick_lens(self) -> str:
        """Pick an abstract lens, avoiding recently used ones."""
        used = set(self.state.get("used_lenses", []))
        unused = [l for l in ABSTRACT_LENSES if l not in used]
        if not unused:
            self.state["used_lenses"] = []
            unused = list(ABSTRACT_LENSES)
        lens = random.choice(unused)
        self.state.setdefault("used_lenses", []).append(lens)
        return lens

    # -- NREM Phase ----------------------------------------------------------

    def nrem_phase(self) -> List[Dict]:
        """
        NREM consolidation: find cross-domain bridges between recently
        modified minds by probing pairs with shared queries and measuring
        semantic similarity of results.
        """
        log.info("=== NREM PHASE: Cross-domain consolidation ===")
        t0 = time.time()

        recent_minds = self._get_recent_minds(hours=24)
        if len(recent_minds) < 2:
            log.info("Not enough minds for consolidation (need at least 2)")
            return []

        mind_names = [m["name"] for m in recent_minds]
        log.info(f"Recent minds ({len(mind_names)}): {', '.join(mind_names)}")

        bridges: List[Dict] = []
        pairs_checked = 0

        for mind_a, mind_b in combinations(mind_names, 2):
            if len(bridges) >= MAX_BRIDGES_PER_CYCLE:
                log.info(f"Bridge limit reached ({MAX_BRIDGES_PER_CYCLE})")
                break

            probe = self._get_probe_query(mind_a, mind_b)
            self.state["probe_index"] += 1

            results_a = self.engine.query(probe, mind_a, top=2)
            results_b = self.engine.query(probe, mind_b, top=2)

            if not results_a or not results_b:
                continue
            if "[error" in results_a or "[error" in results_b:
                continue

            pairs_checked += 1

            text_a = results_a[:500]
            text_b = results_b[:500]
            sim = self.engine.cosine_similarity(text_a, text_b)

            log.info(
                f"  {mind_a} <-> {mind_b}: similarity={sim:.3f} "
                f"(probe: {probe[:60]}...)"
            )

            if sim >= SIMILARITY_THRESHOLD:
                bridge = {
                    "mind_a": mind_a,
                    "mind_b": mind_b,
                    "similarity": round(sim, 4),
                    "probe": probe,
                    "excerpt_a": text_a[:300],
                    "excerpt_b": text_b[:300],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                bridges.append(bridge)
                log.info(
                    f"  ** BRIDGE FOUND: {mind_a} <-> {mind_b} "
                    f"(sim={sim:.3f})"
                )

        # Optionally ingest bridges into a dedicated mind
        if bridges:
            bridge_entries = []
            for b in bridges:
                text = (
                    f"Cross-domain bridge: {b['mind_a']} <-> {b['mind_b']} "
                    f"(similarity: {b['similarity']})\n"
                    f"Probe: {b['probe']}\n"
                    f"From {b['mind_a']}: {b['excerpt_a']}\n"
                    f"From {b['mind_b']}: {b['excerpt_b']}"
                )
                bridge_entries.append(
                    {
                        "text": text,
                        "source": "dream_nrem",
                        "mind_a": b["mind_a"],
                        "mind_b": b["mind_b"],
                        "similarity": b["similarity"],
                        "timestamp": b["timestamp"],
                    }
                )
            ingested = ingest_to_mind(
                "cross_domain_bridges", bridge_entries, self.minds_dir, self.engine
            )
            self.state["bridges_found"] += ingested

        self.state["last_bridges"] = bridges

        elapsed = time.time() - t0
        log.info(
            f"NREM complete: {pairs_checked} pairs checked, "
            f"{len(bridges)} bridges found, {elapsed:.1f}s"
        )
        return bridges

    # -- REM Phase -----------------------------------------------------------

    def rem_phase(self, bridges: Optional[List[Dict]] = None) -> List[Dict]:
        """
        REM hypothesis generation: use LLM to synthesize novel insights
        from cross-domain bridges.
        """
        log.info("=== REM PHASE: Hypothesis generation ===")
        t0 = time.time()

        if bridges is None:
            bridges = self.state.get("last_bridges", [])

        if not bridges:
            log.info("No bridges available for hypothesis generation")
            return []

        hypotheses: List[Dict] = []

        for bridge in bridges[:MAX_HYPOTHESES_PER_CYCLE]:
            mind_a = bridge["mind_a"]
            mind_b = bridge["mind_b"]
            excerpt_a = bridge.get("excerpt_a", "")[:400]
            excerpt_b = bridge.get("excerpt_b", "")[:400]

            prompt = (
                f"You are a cross-domain synthesis engine. Two seemingly "
                f"unrelated knowledge domains share a hidden connection.\n\n"
                f"DOMAIN A ({mind_a}):\n{excerpt_a}\n\n"
                f"DOMAIN B ({mind_b}):\n{excerpt_b}\n\n"
                f"Given that these concepts from '{mind_a.replace('_', ' ')}' "
                f"relate to concepts from '{mind_b.replace('_', ' ')}', "
                f"generate ONE novel hypothesis or technique that could "
                f"emerge from this connection.\n\n"
                f"Your hypothesis should be:\n"
                f"1. Specific and actionable (not vague)\n"
                f"2. Novel (not obvious from either domain alone)\n"
                f"3. Potentially useful for system design, optimization, "
                f"or problem-solving\n\n"
                f"Format your response as:\n"
                f"HYPOTHESIS: [one clear sentence]\n"
                f"REASONING: [2-3 sentences explaining the cross-domain insight]\n"
                f"APPLICATION: [how this could be applied practically]\n"
                f"DOMAINS: [comma-separated relevant domains]\n"
            )

            response = _llm_call(
                prompt,
                ollama_url=self.ollama_url,
                model=self.model,
                temperature=0.8,
                max_tokens=500,
            )
            if not response:
                continue

            hypothesis = self._parse_hypothesis(response, mind_a, mind_b)
            if hypothesis:
                hypothesis["novelty_score"] = self._score_novelty(
                    hypothesis["hypothesis"]
                )
                hypothesis["actionability"] = self._score_actionability(
                    hypothesis
                )
                hypothesis["relevance"] = self._score_relevance(
                    hypothesis.get("domains", []), mind_a, mind_b
                )
                hypothesis["combined_score"] = round(
                    0.35 * hypothesis["novelty_score"]
                    + 0.35 * hypothesis["actionability"]
                    + 0.30 * hypothesis["relevance"],
                    3,
                )

                hypotheses.append(hypothesis)
                log.info(
                    f"  Hypothesis: {hypothesis['hypothesis'][:80]}... "
                    f"(score={hypothesis['combined_score']})"
                )

        hypotheses.sort(key=lambda h: h["combined_score"], reverse=True)

        # Write to output
        for h in hypotheses:
            self._write_insight(
                {
                    "type": "hypothesis",
                    "mind_a": h["mind_a"],
                    "mind_b": h["mind_b"],
                    "hypothesis": h["hypothesis"],
                    "reasoning": h.get("reasoning", ""),
                    "application": h.get("application", ""),
                    "domains": h.get("domains", []),
                    "novelty_score": h["novelty_score"],
                    "actionability": h["actionability"],
                    "relevance": h["relevance"],
                    "combined_score": h["combined_score"],
                    "source": "dream_rem",
                }
            )

        # Ingest into hypotheses mind
        if hypotheses:
            hyp_entries = []
            for h in hypotheses:
                text = (
                    f"Hypothesis: {h['hypothesis']}\n"
                    f"Reasoning: {h.get('reasoning', '')}\n"
                    f"Application: {h.get('application', '')}\n"
                    f"Source domains: {h['mind_a']} + {h['mind_b']}\n"
                    f"Score: {h['combined_score']} "
                    f"(novelty={h['novelty_score']}, "
                    f"actionability={h['actionability']}, "
                    f"relevance={h['relevance']})"
                )
                hyp_entries.append(
                    {
                        "text": text,
                        "source": "dream_rem",
                        "mind_a": h["mind_a"],
                        "mind_b": h["mind_b"],
                        "score": h["combined_score"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            ingested = ingest_to_mind(
                "hypotheses", hyp_entries, self.minds_dir, self.engine
            )
            self.state["hypotheses_generated"] += ingested

        elapsed = time.time() - t0
        log.info(
            f"REM complete: {len(hypotheses)} hypotheses generated, "
            f"{elapsed:.1f}s"
        )
        return hypotheses

    # -- Parliament Mode (full debate) ----------------------------------------

    def parliament_cycle(self) -> dict:
        """
        Full parliament cycle with multi-round debate:
        1. Select a lens (abstract question about systems)
        2. Query all available minds in parallel
        3. Detect coalitions and dissent
        4. Run multi-round debate between dissenting experts
        5. Synthesize final insight via LLM
        6. Check novelty, score, and record
        """
        log.info("=" * 60)
        log.info("PARLIAMENT CYCLE: Full multi-round expert debate")
        log.info("=" * 60)
        t0 = time.time()

        all_minds = self.engine.get_mind_names()
        if len(all_minds) < 3:
            log.info("Not enough minds for parliament (need at least 3)")
            return {"verdict": "SKIP", "reason": "insufficient minds"}

        lens = self._pick_lens()
        log.info(f"Lens: {lens}")

        # Phase 1: Consult experts
        log.info("Phase 1: CONSULT (parallel experts)")
        mind_results = self.experts.consult_experts(
            lens, all_minds, top_per_mind=2
        )
        responding = list(mind_results.keys())
        log.info(f"  {len(responding)} minds responded")

        if len(responding) < 2:
            log.info("  Too few experts responded -- skipping")
            return {"verdict": "SKIP", "reason": "insufficient responses"}

        # Phase 2: Detect dissent and coalitions
        log.info("Phase 2: DISSENT + COALITION detection")
        dissent_pairs = self.experts.find_dissent(mind_results, lens)
        coalitions = self.coalition_detector.detect_coalitions(
            mind_results, self.engine
        )

        if dissent_pairs:
            log.info(
                f"  Top dissent: {dissent_pairs[0][0]} vs "
                f"{dissent_pairs[0][1]} (sim: {dissent_pairs[0][2]:.3f})"
            )
        if coalitions:
            log.info(f"  Coalitions detected: {len(coalitions)}")
            for i, c in enumerate(coalitions[:3]):
                log.info(f"    Faction {i + 1}: {', '.join(c[:4])}")

        # Phase 3: Deliberate
        log.info("Phase 3: DELIBERATE (consensus measurement)")
        deliberation = self.parliament.deliberate(
            lens, mind_results, dissent_pairs, self.engine
        )
        log.info(
            f"  Consensus: {deliberation['consensus_strength']:.0%} "
            f"(avg similarity: {deliberation['avg_similarity']:.3f})"
        )

        # Phase 4: Synthesize debate text
        log.info("Phase 4: SYNTHESIZE (debate prompt)")
        debate_text = self.parliament.synthesize_debate(
            mind_results, dissent_pairs, lens
        )

        prompt = f"""Lens: "{lens}"

{debate_text}

Parliament consensus: {deliberation['consensus_strength']:.0%}

1. Find the deep MECHANISM shared across 2+ domains.
2. Where experts DISAGREE -- what does the gap reveal?
3. What specific, actionable technique emerges?

PATTERN: [shared mechanism, 1 sentence]
TRANSFER: [what one domain teaches another]
DISSENT_INSIGHT: [what the disagreement reveals]
APPLICATION: [specific practical technique]

If no real pattern, say NO_PATTERN_FOUND."""

        synthesis = _llm_call(
            prompt,
            ollama_url=self.ollama_url,
            model=self.model,
            temperature=0.7,
            max_tokens=800,
            timeout=120,
        )

        if not synthesis or "NO_PATTERN" in synthesis:
            log.info("  No pattern found in initial synthesis")
            for a in responding[:3]:
                for b in responding[:3]:
                    if a != b:
                        self.schema.weaken(a, b, 0.15)
            self.reputation.record_participation(responding, "ARCHIVE")
            return {"verdict": "ARCHIVE", "reason": "no pattern"}

        # Phase 5: Multi-round debate
        if dissent_pairs:
            log.info("Phase 5: MULTI-ROUND DEBATE")
            dissenting_minds = [
                dissent_pairs[0][0],
                dissent_pairs[0][1],
            ]
            if len(dissent_pairs) > 1:
                dissenting_minds.extend(
                    [dissent_pairs[1][0], dissent_pairs[1][1]]
                )
            dissenting_minds = list(set(dissenting_minds))

            synthesis = self.parliament.multi_round_debate(
                synthesis,
                dissenting_minds,
                mind_results,
                self.engine,
                ollama_url=self.ollama_url,
                model=self.model,
                rounds=2,
            )

        # Phase 6: Novelty check
        log.info("Phase 6: NOVELTY check")
        is_novel, nearest_sim, nearest_text = self.novelty.check_novelty(
            synthesis, self.engine
        )
        if not is_novel and nearest_sim > 0.90:
            log.info(
                f"  STALE (sim={nearest_sim:.2f}): {nearest_text[:60]}"
            )
            return {"verdict": "ARCHIVE", "reason": "stale insight"}
        if not is_novel:
            log.info(
                f"  Similar ({nearest_sim:.2f}) but proceeding: "
                f"{nearest_text[:60]}"
            )

        # Phase 7: Score
        log.info("Phase 7: SCORE")
        has_pattern = any(
            kw in synthesis.lower()
            for kw in ["pattern", "mechanism", "transfer", "application"]
        )
        has_substance = len(synthesis) > 150
        verdict = (
            "ACTIONABLE" if has_pattern and has_substance else "INTERESTING"
        )
        novelty_score = round(1 - nearest_sim, 3)

        log.info(f"  Verdict: {verdict} | Novelty: {novelty_score}")
        log.info(f"  Synthesis: {synthesis[:200]}...")

        # Phase 8: Consolidate
        log.info("Phase 8: CONSOLIDATE")

        if verdict in ("ACTIONABLE", "INTERESTING"):
            # Hebbian: strengthen connections between responding minds
            for i, a in enumerate(responding[:5]):
                for b in responding[i + 1 : 5]:
                    self.schema.strengthen(a, b, 1.0)

            self.reputation.record_participation(responding, verdict)
            self.novelty.record_insight(synthesis, self.engine)
            self.state["insights_found"] = (
                self.state.get("insights_found", 0) + 1
            )

            insight_entry = {
                "type": "parliament_insight",
                "lens": lens,
                "minds": responding,
                "mind_count": len(responding),
                "synthesis": synthesis[:3000],
                "verdict": verdict,
                "consensus": deliberation["consensus_strength"],
                "novelty_score": novelty_score,
                "dissent_count": len(dissent_pairs),
                "coalitions": len(coalitions),
                "source": "parliament",
            }
            self._write_insight(insight_entry)

            # Ingest the synthesized insight back
            insight_text = (
                f"Parliament insight ({', '.join(responding[:4])}): "
                f"{synthesis[:500]}"
            )
            ingest_to_mind(
                "parliament_insights",
                [
                    {
                        "text": insight_text,
                        "source": "parliament",
                        "minds": responding,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
                self.minds_dir,
                self.engine,
            )
        else:
            for a in responding[:3]:
                for b in responding[:3]:
                    if a != b:
                        self.schema.weaken(a, b, 0.15)
            self.reputation.record_participation(responding, verdict)

        top_edges = self.schema.strongest_connections(3)
        if top_edges:
            log.info(
                f"  Strongest edges: "
                f"{', '.join(f'{k}={v:.1f}' for k, v in top_edges[:3])}"
            )

        elapsed = time.time() - t0
        log.info(
            f"Parliament complete: verdict={verdict}, "
            f"novelty={novelty_score}, {elapsed:.1f}s"
        )

        return {
            "verdict": verdict,
            "synthesis": synthesis[:500],
            "novelty_score": novelty_score,
            "consensus": deliberation["consensus_strength"],
            "minds": responding,
            "elapsed": round(elapsed, 1),
        }

    # -- Full dream cycle (NREM + REM) ----------------------------------------

    def dream(
        self,
        nrem_only: bool = False,
        rem_only: bool = False,
        parliament: bool = False,
    ) -> Dict:
        """Run a full dream cycle."""
        t0 = time.time()
        log.info("=" * 60)
        log.info("DREAM CYCLE STARTING")
        log.info("=" * 60)

        result: Dict = {
            "bridges": [],
            "hypotheses": [],
            "parliament": None,
            "bridges_found": 0,
            "hypotheses_generated": 0,
        }

        if parliament:
            parliament_result = self.parliament_cycle()
            result["parliament"] = parliament_result
        else:
            # Phase 1: NREM
            if not rem_only:
                bridges = self.nrem_phase()
                result["bridges"] = bridges
                result["bridges_found"] = len(bridges)
            else:
                bridges = None

            # Phase 2: REM
            if not nrem_only:
                hypotheses = self.rem_phase(bridges)
                result["hypotheses"] = hypotheses
                result["hypotheses_generated"] = len(hypotheses)

        self.state["total_dreams"] += 1
        self.state["last_dream"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

        elapsed = time.time() - t0
        log.info("=" * 60)
        log.info(
            f"DREAM CYCLE COMPLETE: {result['bridges_found']} bridges, "
            f"{result['hypotheses_generated']} hypotheses, {elapsed:.1f}s"
        )
        log.info("=" * 60)

        return result

    # -- Hypothesis parsing helpers -------------------------------------------

    def _parse_hypothesis(
        self, response: str, mind_a: str, mind_b: str
    ) -> Optional[Dict]:
        try:
            result = {
                "mind_a": mind_a,
                "mind_b": mind_b,
                "hypothesis": "",
                "reasoning": "",
                "application": "",
                "domains": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            hyp_match = re.search(
                r"HYPOTHESIS:\s*(.+?)(?=\n(?:REASONING|APPLICATION|DOMAINS|$))",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if hyp_match:
                result["hypothesis"] = hyp_match.group(1).strip()

            reason_match = re.search(
                r"REASONING:\s*(.+?)(?=\n(?:APPLICATION|DOMAINS|$))",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if reason_match:
                result["reasoning"] = reason_match.group(1).strip()

            app_match = re.search(
                r"APPLICATION:\s*(.+?)(?=\n(?:DOMAINS|$))",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if app_match:
                result["application"] = app_match.group(1).strip()

            domain_match = re.search(
                r"DOMAINS:\s*(.+)", response, re.IGNORECASE
            )
            if domain_match:
                result["domains"] = [
                    d.strip()
                    for d in domain_match.group(1).split(",")
                    if d.strip()
                ]

            if not result["hypothesis"]:
                result["hypothesis"] = response.strip()[:300]

            return result
        except Exception as e:
            log.warning(f"Failed to parse hypothesis: {e}")
            return None

    def _score_novelty(self, hypothesis_text: str) -> float:
        """Score novelty by checking similarity to existing knowledge."""
        if not hypothesis_text:
            return 0.0
        is_novel, sim, _ = self.novelty.check_novelty(
            hypothesis_text, self.engine
        )
        if is_novel:
            return round(max(0.0, 1.0 - sim), 3)
        return round(max(0.0, 1.0 - sim), 3)

    def _score_actionability(self, hypothesis: dict) -> float:
        """Score how actionable a hypothesis is."""
        score = 0.3  # base
        text = (
            hypothesis.get("hypothesis", "")
            + " "
            + hypothesis.get("application", "")
        ).lower()

        action_words = [
            "implement", "build", "create", "detect", "monitor",
            "apply", "use", "combine", "integrate", "test",
            "measure", "optimize", "reduce", "increase",
        ]
        hits = sum(1 for w in action_words if w in text)
        score += min(0.5, hits * 0.1)

        if hypothesis.get("application") and len(hypothesis["application"]) > 30:
            score += 0.2

        return round(min(1.0, score), 3)

    def _score_relevance(
        self, domains: List[str], mind_a: str, mind_b: str
    ) -> float:
        """Score relevance based on domain tags."""
        if not domains:
            return 0.3
        domain_set = {d.strip().lower().replace(" ", "_") for d in domains}
        domain_set.add(mind_a.lower())
        domain_set.add(mind_b.lower())
        overlap = domain_set & RELEVANCE_TAGS
        if overlap:
            return min(1.0, 0.4 + 0.15 * len(overlap))
        return 0.3


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-domain knowledge synthesis engine. "
            "Finds hidden connections between FAISS knowledge bases "
            "by having expert perspectives debate each other."
        )
    )
    parser.add_argument(
        "--minds-dir",
        required=True,
        help="Directory containing mind subdirectories (each with index.faiss)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of dream cycles to run (default: 1)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file path (default: <minds-dir>/dream_insights.jsonl)",
    )
    parser.add_argument(
        "--nrem-only",
        action="store_true",
        help="Run NREM consolidation phase only (find bridges, no LLM)",
    )
    parser.add_argument(
        "--rem-only",
        action="store_true",
        help="Run REM hypothesis phase only (uses cached bridges)",
    )
    parser.add_argument(
        "--parliament",
        action="store_true",
        help="Run parliament mode with full multi-round expert debate",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current dream engine state and exit",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Loop interval in seconds (0 = run once and exit)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [dream] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    engine = DreamEngine(
        minds_dir=args.minds_dir,
        ollama_url=args.ollama_url,
        model=args.model,
        output_path=args.output,
    )

    if args.status:
        state = engine.state.copy()
        if "last_bridges" in state:
            state["last_bridges_count"] = len(state["last_bridges"])
            del state["last_bridges"]
        state["available_minds"] = engine.engine.get_mind_names()
        print(json.dumps(state, indent=2))
        return

    if args.loop > 0:
        log.info(
            f"Starting dream loop (interval={args.loop}s, "
            f"model={args.model})"
        )
        while True:
            try:
                engine.dream(
                    nrem_only=args.nrem_only,
                    rem_only=args.rem_only,
                    parliament=args.parliament,
                )
            except Exception as e:
                log.error(f"Dream cycle failed: {e}", exc_info=True)
            log.info(f"Sleeping {args.loop}s until next dream...")
            time.sleep(args.loop)
    else:
        for cycle_num in range(args.cycles):
            if args.cycles > 1:
                log.info(
                    f"--- Cycle {cycle_num + 1}/{args.cycles} ---"
                )
            result = engine.dream(
                nrem_only=args.nrem_only,
                rem_only=args.rem_only,
                parliament=args.parliament,
            )
            print(f"Bridges found: {result['bridges_found']}")
            print(f"Hypotheses generated: {result['hypotheses_generated']}")
            if result.get("parliament"):
                p = result["parliament"]
                print(
                    f"Parliament verdict: {p.get('verdict', 'N/A')}"
                )
                if p.get("synthesis"):
                    print(f"Synthesis: {p['synthesis'][:200]}...")
            print(
                f"Total insights: "
                f"{engine.state.get('insights_found', 0)}"
            )


if __name__ == "__main__":
    main()
