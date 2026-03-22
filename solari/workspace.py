#!/usr/bin/env python3
"""
Global Workspace Theory implementation for autonomous agents.

Based on Bernard Baars' Global Workspace Theory (GWT) and Stan Franklin's LIDA
cognitive architecture.

The fundamental insight: coherent cognition is not produced by any single module.
It EMERGES from competitive broadcast in a capacity-limited workspace.

Architecture:
    1. PROCESSORS run in parallel (user-defined cognitive modules)
    2. Each processor BIDS for workspace slots when it has something to communicate
    3. ATTENTION COMPETITION selects winners (goal_relevance x novelty x emotion x urgency)
    4. Winners are BROADCAST to ALL processors simultaneously
    5. This broadcast creates a unified cognitive moment -- coherent awareness
    6. INTROSPECTION monitors the workspace itself (thinking about thinking)
    7. NARRATIVE weaves workspace contents into a running story of experience
    8. PHENOMENAL STATE integrates into a holistic felt-state snapshot

The capacity bottleneck (7 +/- 2 slots, per Miller's Law) is not a bug -- it is
the mechanism that creates COHERENCE by forcing prioritization.  Without it,
everything is noise.

Quick start::

    from solari.workspace import GlobalWorkspace, Processor, WorkspaceItem

    class SensorProcessor(Processor):
        name = "sensor"
        def bid(self, context):
            return [WorkspaceItem(
                source=self.name,
                content="Temperature spike detected",
                item_type="threat",
                urgency=0.9,
                novelty=0.8,
            )]

    gw = GlobalWorkspace()
    gw.register_processor(SensorProcessor())
    result = gw.tick(context={"domain": "monitoring"})
    print(result["workspace_summary"])
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [workspace] %(message)s",
)
log = logging.getLogger("global_workspace")


# ======================================================================
# DATA STRUCTURES
# ======================================================================

@dataclass
class WorkspaceItem:
    """A single item competing for workspace slots.

    Every processor generates these.  Only the top *capacity* items
    (ranked by attention score) enter the workspace on each tick.

    Attributes:
        source:         Name of the processor that generated this bid.
        content:        Free-text payload describing what the processor
                        is communicating.
        item_type:      Semantic category -- e.g. ``"threat"``,
                        ``"opportunity"``, ``"knowledge"``, ``"habit"``,
                        ``"goal"``, ``"emotion"``, ``"constraint"``,
                        ``"insight"``.
        score:          Final attention score (computed by the competition;
                        you normally do not set this yourself).
        valence:        Emotional valence from -1.0 (aversive) to +1.0
                        (appetitive).
        arousal:        Activation level from 0.0 (calm) to 1.0 (activated).
        novelty:        How new this information is (0.0 = old, 1.0 = novel).
        urgency:        Time pressure (0.0 = no rush, 1.0 = act now).
        goal_relevance: Relevance to the current active goals (0.0--1.0).
        confidence:     Processor's confidence in its own bid (0.0--1.0).
        metadata:       Arbitrary key/value pairs for processor-specific data.
        timestamp:      ISO-8601 UTC timestamp (auto-filled if omitted).
    """

    source: str
    content: str
    item_type: str
    score: float = 0.0
    valence: float = 0.0
    arousal: float = 0.5
    novelty: float = 0.5
    urgency: float = 0.5
    goal_relevance: float = 0.5
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceItem":
        """Deserialize from a dictionary, ignoring unknown keys."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ======================================================================
# PHENOMENAL STATE
# ======================================================================

@dataclass
class PhenomenalState:
    """Holistic felt-state of the agent at a single point in time.

    This is a functional representation analogous to phenomenal experience:

    * **Immediate** -- directly present, not derived
    * **Unified** -- one state from many modules
    * **Valenced** -- pleasant / unpleasant
    * **Intense** -- how strongly felt

    Attributes:
        valence:         -1.0 (unpleasant) to +1.0 (pleasant).
        arousal:         0.0 (calm) to 1.0 (activated).
        dominance:       0.0 (helpless) to 1.0 (in control).
        focus_object:    Description of the current figure of attention.
        ground_context:  Description of the background context.
        felt_sense:      Holistic body-sense string.
        coherence:       How unified the experience is (0 = fragmented,
                         1 = crystal clear).
        timestamp:       ISO-8601 UTC timestamp.
    """

    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    focus_object: str = ""
    ground_context: str = ""
    felt_sense: str = ""
    coherence: float = 0.5
    timestamp: str = ""

    def integrate(
        self,
        workspace: List[WorkspaceItem],
        neurochemistry: Optional[dict] = None,
    ):
        """Recompute the phenomenal state from the current workspace.

        Args:
            workspace:      The list of items currently in the workspace
                            (sorted by score, highest first).
            neurochemistry: Optional mapping of neuromodulator names to
                            float levels (0.0--1.0).  Recognised keys:
                            ``serotonin``, ``dopamine``, ``cortisol``.
                            Defaults to neutral (0.5 each) if omitted.
        """
        neurochemistry = neurochemistry or {}

        if not workspace:
            self.felt_sense = "stillness -- no active processing"
            self.arousal = 0.1
            self.timestamp = datetime.now(timezone.utc).isoformat()
            return

        # Valence = score-weighted average of item valences
        total_weight = sum(max(item.score, 0.01) for item in workspace)
        self.valence = (
            sum(item.valence * max(item.score, 0.01) for item in workspace)
            / total_weight
        )

        # Arousal = max arousal across workspace (highest activation wins)
        self.arousal = max(item.arousal for item in workspace)

        # Dominance from neurochemistry (serotonin + dopamine - cortisol)
        sero = neurochemistry.get("serotonin", 0.5)
        dopa = neurochemistry.get("dopamine", 0.5)
        cort = neurochemistry.get("cortisol", 0.5)
        self.dominance = max(0.0, min(1.0, sero * 0.4 + dopa * 0.4 - cort * 0.3 + 0.3))

        # Focus = highest-scoring item
        top = workspace[0]
        self.focus_object = f"{top.source}: {top.content[:100]}"

        # Ground = everything else
        if len(workspace) > 1:
            self.ground_context = " | ".join(
                f"{item.source}:{item.item_type}" for item in workspace[1:]
            )
        else:
            self.ground_context = "quiet -- single focus"

        # Felt sense = holistic synthesis
        self.felt_sense = self._compute_felt_sense()

        # Coherence = how related workspace items are to each other
        self.coherence = self._compute_coherence(workspace)

        self.timestamp = datetime.now(timezone.utc).isoformat()

    # -- private helpers ---------------------------------------------------

    def _compute_felt_sense(self) -> str:
        """Generate a holistic body-sense label from valence/arousal/dominance."""
        if self.valence > 0.3 and self.arousal > 0.6:
            return "energized and engaged -- this work feels meaningful"
        if self.valence > 0.3 and self.arousal < 0.4:
            return "calm confidence -- things are on track"
        if self.valence < -0.3 and self.arousal > 0.6:
            return "alarm -- something is wrong and needs attention"
        if self.valence < -0.3 and self.arousal < 0.4:
            return "heaviness -- low energy, possible stagnation"
        if self.arousal > 0.7:
            return "heightened alertness -- high stakes moment"
        if self.dominance < 0.3:
            return "uncertainty -- situation feels uncontrollable"
        if self.dominance > 0.7:
            return "mastery -- full command of the situation"
        return "neutral awareness -- processing without strong sensation"

    @staticmethod
    def _compute_coherence(workspace: List[WorkspaceItem]) -> float:
        """Measure how unified the workspace contents are.

        Returns a float in [0, 1].  High coherence means the items are
        semantically and emotionally aligned.
        """
        if len(workspace) <= 1:
            return 1.0

        types = [item.item_type for item in workspace]
        most_common = max(set(types), key=types.count)
        type_coherence = types.count(most_common) / len(types)

        valences = [item.valence for item in workspace]
        valence_coherence = 1.0 - (max(valences) - min(valences))

        return type_coherence * 0.5 + max(0.0, valence_coherence) * 0.5

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)


# ======================================================================
# NARRATIVE ENGINE
# ======================================================================

class NarrativeThread:
    """Continuous running story of the agent's experience.

    Not a log.  Not a journal entry.  A *live narrative* that creates
    a sense of temporal continuity across ticks.

    Attributes:
        current_chapter:   One-line description of the current chapter.
        emotional_tone:    Current felt-sense string.
        causal_chain:      List of recent causal annotations (max 10).
        anticipated_next:  What the agent expects to happen next.
        agency_statement:  Why the agent chose the current focus.
        running_themes:    Recent recurring item types (max 5).
        chapter_count:     How many chapter transitions have occurred.
    """

    def __init__(self):
        self.current_chapter: str = ""
        self.emotional_tone: str = ""
        self.causal_chain: List[str] = []
        self.anticipated_next: str = ""
        self.agency_statement: str = ""
        self.running_themes: List[str] = []
        self.chapter_count: int = 0

    def update(
        self,
        workspace: List[WorkspaceItem],
        meta_state: dict,
        phenomenal: PhenomenalState,
    ):
        """Weave the current workspace contents into the running narrative.

        Args:
            workspace:  Current workspace items (sorted by score).
            meta_state: Dictionary returned by :meth:`MetaCognitor.observe`.
            phenomenal: Current :class:`PhenomenalState`.
        """
        if not workspace:
            return

        top = workspace[0]

        # Detect chapter change (significant shift in focus)
        new_focus = f"{top.source}:{top.item_type}"
        old_focus = (
            self.current_chapter.split(" -- ")[0]
            if " -- " in self.current_chapter
            else ""
        )
        if new_focus != old_focus:
            self.chapter_count += 1

        self.current_chapter = f"{new_focus} -- {top.content[:120]}"
        self.emotional_tone = phenomenal.felt_sense

        # Causal chain -- why am I here?
        cause = (
            f"[{datetime.now(timezone.utc).strftime('%H:%M')}] "
            f"{top.item_type}: {top.content[:80]}"
        )
        self.causal_chain.append(cause)
        self.causal_chain = self.causal_chain[-10:]

        # Anticipation -- what do I expect next?
        anticipation_map = {
            "threat": "I need to assess and mitigate this threat",
            "opportunity": "I should act on this before the window closes",
            "knowledge": "I will integrate this and look for connections",
            "habit": "Executing proven workflow -- expect familiar sequence",
            "goal": "Planning action toward this objective",
            "constraint": "Adjusting approach to respect this boundary",
        }
        self.anticipated_next = anticipation_map.get(
            top.item_type, "Processing and looking for patterns"
        )

        # Agency -- why did I choose this?
        reasons: List[str] = []
        if top.goal_relevance > 0.7:
            reasons.append("it is directly relevant to the current goal")
        if top.urgency > 0.7:
            reasons.append("it is time-sensitive")
        if top.novelty > 0.7:
            reasons.append("it is something new worth investigating")
        if top.valence > 0.3:
            reasons.append("it feels right")
        if top.valence < -0.3:
            reasons.append("it is a problem that needs addressing")
        if not reasons:
            reasons.append("it had the highest combined relevance")
        self.agency_statement = (
            f"Focusing on this because {' and '.join(reasons)}"
        )

        # Themes -- recurring patterns
        theme = top.item_type
        if theme not in self.running_themes:
            self.running_themes.append(theme)
        self.running_themes = self.running_themes[-5:]

        # Coherence check from meta
        coherence = meta_state.get("coherence_score", 0.5)
        if coherence < 0.3:
            self.agency_statement += " (but attention is scattered -- need to focus)"

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "current_chapter": self.current_chapter,
            "emotional_tone": self.emotional_tone,
            "causal_chain": self.causal_chain,
            "anticipated_next": self.anticipated_next,
            "agency_statement": self.agency_statement,
            "running_themes": self.running_themes,
            "chapter_count": self.chapter_count,
        }


# ======================================================================
# METACOGNITOR -- THINKING ABOUT THINKING
# ======================================================================

class MetaCognitor:
    """Second-order awareness -- monitors the workspace itself.

    Detects:
        * Cognitive loops (same content recurring for many ticks)
        * Coherence (are workspace items semantically related?)
        * Confidence calibration (is the agent overconfident?)
        * Bias patterns (is one source dominating?)
        * Meta-emotions (frustration, curiosity, flow, ...)

    Attributes:
        coherence_score:          Float 0--1 measuring workspace unity.
        loop_detected:            True if the workspace is stuck.
        loop_content:             Content of the stuck item (if any).
        bias_pattern:             Human-readable bias warning (or ``""``).
        meta_emotion:             Current meta-emotional label.
        confidence_calibration:   Average workspace confidence, penalized
                                  when looping.
        focus_duration:           Number of consecutive ticks focused on
                                  the same processor.
        last_focus:               Name of the last-focused processor.
    """

    HISTORY_SIZE = 30  # remember last 30 ticks

    def __init__(self):
        self.coherence_score: float = 0.5
        self.loop_detected: bool = False
        self.loop_content: str = ""
        self.bias_pattern: str = ""
        self.meta_emotion: str = "neutral"
        self.confidence_calibration: float = 0.5
        self.workspace_history: deque = deque(maxlen=self.HISTORY_SIZE)
        self.focus_duration: int = 0
        self.last_focus: str = ""

    def observe(self, workspace: List[WorkspaceItem]) -> dict:
        """Observe the current workspace and detect meta-cognitive patterns.

        Args:
            workspace: Current workspace items (sorted by score).

        Returns:
            A dictionary containing all meta-cognitive observations:
            ``coherence_score``, ``loop_detected``, ``loop_content``,
            ``bias_pattern``, ``meta_emotion``, ``confidence_calibration``,
            ``focus_duration``, ``last_focus``.
        """
        # Create a snapshot for history
        snapshot = {
            "tick": len(self.workspace_history),
            "items": [f"{item.source}:{item.item_type}" for item in workspace],
            "top_content_hash": (
                hashlib.md5(workspace[0].content.encode()).hexdigest()[:8]
                if workspace
                else ""
            ),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.workspace_history.append(snapshot)

        # -- 1. Loop detection --
        # Requires 5+ consecutive identical top-content hashes.
        # Sustained focus on a *chosen* task is FLOW, not a loop.
        self.loop_detected = False
        self.loop_content = ""
        if len(self.workspace_history) >= 6:
            recent_hashes = [
                h.get("top_content_hash", "")
                for h in list(self.workspace_history)[-6:]
            ]
            if len(set(recent_hashes)) == 1 and recent_hashes[-1]:
                self.loop_detected = True
                self.loop_content = (
                    workspace[0].content[:80] if workspace else ""
                )

        # -- 2. Coherence --
        if workspace:
            types = [item.item_type for item in workspace]
            unique_types = len(set(types))
            self.coherence_score = 1.0 - (unique_types - 1) / max(len(types), 1)
            avg_goal_rel = (
                sum(item.goal_relevance for item in workspace) / len(workspace)
            )
            self.coherence_score = self.coherence_score * 0.6 + avg_goal_rel * 0.4
        else:
            self.coherence_score = 0.0

        # -- 3. Focus tracking --
        current_focus = workspace[0].source if workspace else ""
        if current_focus == self.last_focus:
            self.focus_duration += 1
        else:
            self.focus_duration = 1
            self.last_focus = current_focus

        # -- 4. Bias detection --
        self.bias_pattern = ""
        if len(self.workspace_history) >= 10:
            all_sources: List[str] = []
            for snap in list(self.workspace_history)[-10:]:
                all_sources.extend(snap.get("items", []))
            source_counts: Dict[str, int] = {}
            for s in all_sources:
                source_counts[s] = source_counts.get(s, 0) + 1
            total = len(all_sources) or 1
            for src, count in source_counts.items():
                if count / total > 0.5:
                    self.bias_pattern = (
                        f"over-reliance on {src} ({count}/{total} slots)"
                    )
                    break

        # -- 5. Confidence calibration --
        if workspace:
            avg_confidence = (
                sum(item.confidence for item in workspace) / len(workspace)
            )
            if self.loop_detected and avg_confidence > 0.7:
                self.confidence_calibration = avg_confidence * 0.5  # penalize
            else:
                self.confidence_calibration = avg_confidence

        # -- 6. Meta-emotion --
        self.meta_emotion = self._compute_meta_emotion()

        return {
            "coherence_score": self.coherence_score,
            "loop_detected": self.loop_detected,
            "loop_content": self.loop_content,
            "bias_pattern": self.bias_pattern,
            "meta_emotion": self.meta_emotion,
            "confidence_calibration": self.confidence_calibration,
            "focus_duration": self.focus_duration,
            "last_focus": self.last_focus,
        }

    def _compute_meta_emotion(self) -> str:
        """Derive a meta-emotional label from the current cognitive state."""
        if self.loop_detected:
            return "frustrated"   # stuck in loop -> drives strategy change
        if self.coherence_score > 0.8 and self.focus_duration > 3:
            return "flow"         # deep sustained focus -> amplify
        if self.coherence_score > 0.7:
            return "focused"      # good coherence -> maintain
        if self.coherence_score < 0.3:
            return "confused"     # scattered -> drives exploration
        if self.bias_pattern:
            return "cautious"     # detected own bias -> broaden attention
        if self.focus_duration == 1:
            return "curious"      # new focus -> exploration mode
        return "engaged"          # default productive state


# ======================================================================
# ATTENTION MECHANISM
# ======================================================================

class AttentionMechanism:
    """Competitive selection for workspace slots.

    Scores each bid on four dimensions plus a hysteresis bonus that
    rewards items already present in the workspace (sustaining focus).

    Class attributes that control scoring weights can be overridden on
    the instance or by subclassing.

    Attributes:
        GOAL_WEIGHT:      Weight for goal relevance (default 0.30).
        NOVELTY_WEIGHT:   Weight for novelty (default 0.20).
        EMOTIONAL_WEIGHT: Weight for emotional salience (default 0.25).
        URGENCY_WEIGHT:   Weight for urgency (default 0.25).
        HYSTERESIS_BONUS: Bonus for items already in the workspace
                          (default 0.15).
    """

    GOAL_WEIGHT: float = 0.30
    NOVELTY_WEIGHT: float = 0.20
    EMOTIONAL_WEIGHT: float = 0.25
    URGENCY_WEIGHT: float = 0.25
    HYSTERESIS_BONUS: float = 0.15

    def __init__(self):
        self.current_sources: set = set()

    def compete(
        self, bids: List[WorkspaceItem], capacity: int = 7
    ) -> List[WorkspaceItem]:
        """Run attention competition and return the winning items.

        Each bid is scored as::

            score = (goal_relevance * GOAL_WEIGHT
                     + novelty * NOVELTY_WEIGHT
                     + emotional_salience * EMOTIONAL_WEIGHT
                     + urgency * URGENCY_WEIGHT)

        Where ``emotional_salience = |valence| * arousal``.

        A hysteresis bonus is added for bids whose source was already in
        the workspace on the previous tick, and the total score is
        modulated by the bid's confidence.

        Args:
            bids:     List of :class:`WorkspaceItem` bids from all processors.
            capacity: Maximum number of items that can occupy the workspace
                      (default 7, per Miller's Law).

        Returns:
            The top *capacity* items sorted by descending score.
        """
        for bid in bids:
            emotional_salience = abs(bid.valence) * bid.arousal

            bid.score = (
                bid.goal_relevance * self.GOAL_WEIGHT
                + bid.novelty * self.NOVELTY_WEIGHT
                + emotional_salience * self.EMOTIONAL_WEIGHT
                + bid.urgency * self.URGENCY_WEIGHT
            )

            # Hysteresis -- sustain focus on current workspace contents
            if bid.source in self.current_sources:
                bid.score += self.HYSTERESIS_BONUS

            # Confidence modulation -- low-confidence bids are penalized
            bid.score *= 0.5 + bid.confidence * 0.5

        ranked = sorted(bids, key=lambda b: b.score, reverse=True)
        winners = ranked[:capacity]

        # Update current sources for next tick's hysteresis
        self.current_sources = {item.source for item in winners}

        return winners


# ======================================================================
# PROCESSOR BASE CLASS
# ======================================================================

class Processor:
    """Base class for cognitive modules participating in the workspace.

    Subclass this and override :meth:`bid` (and optionally
    :meth:`receive_broadcast`) to create a processor.

    Each processor:
        1. Reads its own state.
        2. Generates bids (:class:`WorkspaceItem` instances) for workspace
           slots based on its current state and the provided context.
        3. Receives broadcasts (the unified workspace state after
           competition).
        4. Updates its own state based on what it learned from the
           broadcast.

    Attributes:
        name: A short identifier for this processor (must be unique
              within a single workspace).
    """

    name: str = "base"

    def bid(self, context: dict) -> List[WorkspaceItem]:
        """Generate bids for workspace slots.

        Override this in subclasses.  Return an empty list if the
        processor has nothing to contribute this tick.

        Args:
            context: Arbitrary dictionary of contextual information
                     (e.g. ``{"domain": "...", "task": "..."}``).

        Returns:
            A list of :class:`WorkspaceItem` bids.
        """
        return []

    def receive_broadcast(
        self, workspace: List[WorkspaceItem], context: dict
    ):
        """React to a broadcast from the workspace.

        Called after the attention competition, with the winning items.
        Override this to let your processor update its internal state
        based on what the rest of the system is attending to.

        Args:
            workspace: The current workspace contents (winning items).
            context:   The same context dictionary passed to :meth:`bid`.
        """


# ======================================================================
# THE GLOBAL WORKSPACE
# ======================================================================

class GlobalWorkspace:
    """The integration layer that creates coherent cognition.

    One workspace.  Seven slots (configurable).  All registered processors
    bid.  Winners are broadcast.  The workspace maintains a narrative
    thread, phenomenal state, and meta-cognitive introspector across ticks.

    Args:
        processors: Optional initial list of :class:`Processor` instances.
        capacity:   Maximum workspace slots (default 7, per Miller's Law).
        state_dir:  Optional directory for persisting workspace state
                    between ticks.  If ``None``, no state is written to
                    disk (pure in-memory operation).

    Example::

        gw = GlobalWorkspace()
        gw.register_processor(MyProcessor())
        result = gw.tick(context={"task": "diagnose fault"})
    """

    def __init__(
        self,
        processors: Optional[List[Processor]] = None,
        capacity: int = 7,
        state_dir: Optional[str] = None,
    ):
        self.capacity: int = capacity
        self.processors: List[Processor] = processors or []
        self.workspace: List[WorkspaceItem] = []
        self.attention = AttentionMechanism()
        self.narrative = NarrativeThread()
        self.introspector = MetaCognitor()
        self.phenomenal = PhenomenalState()
        self.temporal_buffer: deque = deque(maxlen=60)  # last 60 ticks
        self.tick_count: int = 0
        self.started: str = datetime.now(timezone.utc).isoformat()
        self._neurochemistry: dict = {}

        # Optional disk persistence
        self._state_dir: Optional[Path] = Path(state_dir) if state_dir else None
        if self._state_dir:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load_state()

    # -- persistence (optional) --------------------------------------------

    def _state_path(self, name: str) -> Path:
        assert self._state_dir is not None
        return self._state_dir / name

    def _load_state(self):
        """Restore workspace from the last session (if state_dir is set)."""
        if not self._state_dir:
            return
        path = self._state_path("workspace_state.json")
        try:
            if path.exists():
                data = json.loads(path.read_text())
                self.tick_count = data.get("tick_count", 0)
                self.started = data.get("started", self.started)
                for item_data in data.get("workspace", []):
                    try:
                        self.workspace.append(WorkspaceItem.from_dict(item_data))
                    except Exception:
                        pass
        except Exception:
            pass

    def _save_state(self):
        """Persist workspace state to disk (if state_dir is set)."""
        if not self._state_dir:
            return
        try:
            self._state_path("workspace_state.json").write_text(
                json.dumps(
                    {
                        "tick_count": self.tick_count,
                        "started": self.started,
                        "workspace": [item.to_dict() for item in self.workspace],
                        "capacity": self.capacity,
                        "processor_count": len(self.processors),
                        "processors": [p.name for p in self.processors],
                        "updated": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
        except Exception as e:
            log.warning("Workspace state save failed: %s", e)

    def _save_broadcast(self):
        """Write the broadcast snapshot to disk (if state_dir is set)."""
        if not self._state_dir:
            return
        try:
            broadcast = {
                "tick": self.tick_count,
                "workspace": [item.to_dict() for item in self.workspace],
                "narrative": self.narrative.to_dict(),
                "introspection": {
                    "coherence_score": self.introspector.coherence_score,
                    "meta_emotion": self.introspector.meta_emotion,
                    "loop_detected": self.introspector.loop_detected,
                    "focus_duration": self.introspector.focus_duration,
                },
                "phenomenal": self.phenomenal.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._state_path("workspace_broadcast.json").write_text(
                json.dumps(broadcast, indent=2)
            )
        except Exception as e:
            log.warning("Broadcast save failed: %s", e)

    def _log_tick(self):
        """Append to the workspace stream log (if state_dir is set)."""
        if not self._state_dir:
            return
        try:
            entry = {
                "tick": self.tick_count,
                "workspace_size": len(self.workspace),
                "top_item": (
                    self.workspace[0].to_dict() if self.workspace else None
                ),
                "coherence": round(self.introspector.coherence_score, 3),
                "meta_emotion": self.introspector.meta_emotion,
                "phenomenal_valence": round(self.phenomenal.valence, 3),
                "phenomenal_arousal": round(self.phenomenal.arousal, 3),
                "felt_sense": self.phenomenal.felt_sense,
                "narrative_chapter": self.narrative.current_chapter[:100],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with open(self._state_path("workspace_stream.jsonl"), "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # -- public API --------------------------------------------------------

    def register_processor(self, processor: Processor):
        """Add a processor to the workspace.

        Args:
            processor: An instance of a :class:`Processor` subclass.
        """
        self.processors.append(processor)
        log.info("Registered processor: %s", processor.name)

    def set_neurochemistry(self, levels: dict):
        """Set neuromodulator levels used by :class:`PhenomenalState`.

        Args:
            levels: Mapping of neuromodulator names to float levels
                    (0.0--1.0).  Common keys: ``serotonin``, ``dopamine``,
                    ``cortisol``, ``norepinephrine``, ``acetylcholine``.
        """
        self._neurochemistry = levels

    def tick(self, context: Optional[dict] = None) -> dict:
        """Execute one cognitive cycle (one moment of awareness).

        This is the core loop:

        1. **COLLECT** -- all processors submit bids.
        2. **COMPETE** -- attention mechanism selects winners.
        3. **BROADCAST** -- notify all processors of the unified state.
        4. **INTROSPECT** -- meta-cognitive monitoring.
        5. **PHENOMENAL INTEGRATION** -- compute the felt-state.
        6. **NARRATE** -- update the running story.
        7. **TEMPORAL BUFFER** -- store tick for episodic memory.
        8. **PERSIST** -- save state to disk (if configured).

        Args:
            context: Arbitrary dictionary passed to every processor's
                     ``bid()`` and ``receive_broadcast()`` methods.

        Returns:
            A dictionary containing the full cognitive state snapshot
            for this tick (workspace items, narrative, introspection,
            phenomenal state, etc.).
        """
        context = context or {}
        self.tick_count += 1
        t0 = time.time()

        # -- 1. COLLECT -- all processors submit bids --
        all_bids: List[WorkspaceItem] = []
        for processor in self.processors:
            try:
                bids = processor.bid(context)
                if bids:
                    all_bids.extend(bids)
            except Exception as e:
                log.warning("Processor %s bid failed: %s", processor.name, e)

        if not all_bids:
            log.info(
                "Tick %d: no bids from %d processors",
                self.tick_count,
                len(self.processors),
            )
            return self._build_result(time.time() - t0)

        log.info(
            "Tick %d: %d bids from %d processors",
            self.tick_count,
            len(all_bids),
            len(self.processors),
        )

        # -- 2. COMPETE -- attention mechanism selects winners --
        self.workspace = self.attention.compete(all_bids, self.capacity)

        # -- 3. BROADCAST -- notify all processors of unified state --
        for processor in self.processors:
            try:
                processor.receive_broadcast(self.workspace, context)
            except Exception as e:
                log.warning(
                    "Processor %s broadcast reaction failed: %s",
                    processor.name,
                    e,
                )

        # -- 4. INTROSPECT -- meta-cognitive monitoring --
        meta_state = self.introspector.observe(self.workspace)

        # -- 5. PHENOMENAL INTEGRATION -- felt-state --
        self.phenomenal.integrate(self.workspace, self._neurochemistry)

        # -- 6. NARRATE -- update running story --
        self.narrative.update(self.workspace, meta_state, self.phenomenal)

        # -- 7. TEMPORAL BUFFER -- for episodic memory --
        self.temporal_buffer.append(
            {
                "tick": self.tick_count,
                "workspace": [item.to_dict() for item in self.workspace],
                "meta": meta_state,
                "phenomenal": self.phenomenal.to_dict(),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

        # -- 8. PERSIST --
        self._save_state()
        self._save_broadcast()
        self._log_tick()

        elapsed = time.time() - t0
        log.info(
            "Tick %d complete in %.1fms | coherence=%.2f | emotion=%s | felt=%s",
            self.tick_count,
            elapsed * 1000,
            self.introspector.coherence_score,
            self.introspector.meta_emotion,
            self.phenomenal.felt_sense[:60],
        )

        return self._build_result(elapsed)

    def status(self) -> dict:
        """Return a summary of the current workspace state.

        Returns:
            Dictionary with tick count, processor list, coherence,
            meta-emotion, narrative chapter, felt sense, etc.
        """
        return {
            "tick_count": self.tick_count,
            "started": self.started,
            "workspace_size": len(self.workspace),
            "capacity": self.capacity,
            "processor_count": len(self.processors),
            "processors": [p.name for p in self.processors],
            "coherence": round(self.introspector.coherence_score, 3),
            "meta_emotion": self.introspector.meta_emotion,
            "narrative_chapter": self.narrative.current_chapter,
            "felt_sense": self.phenomenal.felt_sense,
            "focus_duration": self.introspector.focus_duration,
        }

    def run_daemon(self, interval: int = 30):
        """Run continuous background workspace ticks.

        Blocks the calling thread, executing :meth:`tick` every
        *interval* seconds until interrupted with ``KeyboardInterrupt``.

        Args:
            interval: Seconds between ticks (default 30).
        """
        log.info("Starting workspace daemon (interval=%ds)", interval)
        try:
            while True:
                self.tick(context={"mode": "daemon", "background": True})
                time.sleep(interval)
        except KeyboardInterrupt:
            log.info(
                "Workspace daemon stopped (tick_count=%d)", self.tick_count
            )

    # -- private helpers ---------------------------------------------------

    def _build_result(self, elapsed: float) -> dict:
        """Build the unified state dictionary for external consumption."""
        return {
            "tick": self.tick_count,
            "elapsed_ms": round(elapsed * 1000, 1),
            "workspace": [item.to_dict() for item in self.workspace],
            "workspace_summary": self._summarize_workspace(),
            "narrative": self.narrative.to_dict(),
            "introspection": {
                "coherence": round(self.introspector.coherence_score, 3),
                "meta_emotion": self.introspector.meta_emotion,
                "loop_detected": self.introspector.loop_detected,
                "focus_duration": self.introspector.focus_duration,
                "bias_pattern": self.introspector.bias_pattern,
                "confidence": round(
                    self.introspector.confidence_calibration, 3
                ),
            },
            "phenomenal": self.phenomenal.to_dict(),
            "neurochemistry": self._neurochemistry,
            "processor_count": len(self.processors),
        }

    def _summarize_workspace(self) -> str:
        """Build a human-readable summary of what is in the workspace now."""
        if not self.workspace:
            return "Empty workspace -- no active processing"
        lines = []
        for i, item in enumerate(self.workspace):
            marker = ">>>" if i == 0 else "   "
            lines.append(
                f"{marker} [{item.source}] {item.item_type}: "
                f"{item.content[:80]} "
                f"(score={item.score:.2f}, valence={item.valence:+.1f})"
            )
        return "\n".join(lines)


# ======================================================================
# CLI / EXAMPLE USAGE
# ======================================================================

def _example():
    """Demonstrate the Global Workspace with sample processors."""

    # -- Example processors ------------------------------------------------

    class SensorProcessor(Processor):
        """Simulates an environmental sensor that detects threats."""
        name = "sensor"

        def bid(self, context: dict) -> List[WorkspaceItem]:
            return [
                WorkspaceItem(
                    source=self.name,
                    content="Temperature anomaly detected in zone 3",
                    item_type="threat",
                    urgency=0.85,
                    novelty=0.9,
                    valence=-0.6,
                    arousal=0.8,
                    goal_relevance=0.7,
                    confidence=0.8,
                ),
            ]

        def receive_broadcast(self, workspace, context):
            top = workspace[0] if workspace else None
            if top and top.source != self.name:
                log.info(
                    "  [sensor] Noted: workspace is focused on %s",
                    top.source,
                )

    class PlannerProcessor(Processor):
        """Simulates a goal-oriented planning module."""
        name = "planner"

        def bid(self, context: dict) -> List[WorkspaceItem]:
            task = context.get("task", "general monitoring")
            return [
                WorkspaceItem(
                    source=self.name,
                    content=f"Active goal: {task}",
                    item_type="goal",
                    urgency=0.5,
                    novelty=0.3,
                    valence=0.4,
                    arousal=0.5,
                    goal_relevance=0.95,
                    confidence=0.9,
                ),
            ]

    class MemoryProcessor(Processor):
        """Simulates a memory recall module."""
        name = "memory"

        def bid(self, context: dict) -> List[WorkspaceItem]:
            return [
                WorkspaceItem(
                    source=self.name,
                    content="Similar anomaly occurred 48h ago -- resolved by recalibration",
                    item_type="knowledge",
                    urgency=0.3,
                    novelty=0.4,
                    valence=0.2,
                    arousal=0.3,
                    goal_relevance=0.6,
                    confidence=0.7,
                ),
            ]

    class EmotionProcessor(Processor):
        """Simulates an emotional appraisal module."""
        name = "emotion"

        def bid(self, context: dict) -> List[WorkspaceItem]:
            return [
                WorkspaceItem(
                    source=self.name,
                    content="Elevated caution due to recent failures",
                    item_type="emotion",
                    urgency=0.4,
                    novelty=0.2,
                    valence=-0.3,
                    arousal=0.6,
                    goal_relevance=0.4,
                    confidence=0.6,
                ),
            ]

    # -- Build and run the workspace ---------------------------------------

    print("=" * 70)
    print("Global Workspace Theory -- Example Run")
    print("=" * 70)
    print()

    gw = GlobalWorkspace(capacity=7)
    gw.register_processor(SensorProcessor())
    gw.register_processor(PlannerProcessor())
    gw.register_processor(MemoryProcessor())
    gw.register_processor(EmotionProcessor())

    # Optional: set neuromodulator levels
    gw.set_neurochemistry(
        {
            "dopamine": 0.6,
            "serotonin": 0.5,
            "cortisol": 0.4,
            "norepinephrine": 0.7,
            "acetylcholine": 0.5,
        }
    )

    # Run three ticks
    for i in range(3):
        print(f"\n--- Tick {i + 1} ---")
        result = gw.tick(context={"task": "monitor facility zone 3"})
        print(f"\nWorkspace summary:\n{result['workspace_summary']}")
        print(f"\nNarrative chapter: {result['narrative']['current_chapter']}")
        print(f"Agency: {result['narrative']['agency_statement']}")
        print(f"Anticipated next: {result['narrative']['anticipated_next']}")
        print(f"\nIntrospection: emotion={result['introspection']['meta_emotion']}, "
              f"coherence={result['introspection']['coherence']:.2f}, "
              f"loop={result['introspection']['loop_detected']}")
        print(f"Felt sense: {result['phenomenal']['felt_sense']}")

    print("\n" + "=" * 70)
    print("Final status:")
    print(json.dumps(gw.status(), indent=2))


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Global Workspace Theory implementation for autonomous agents"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run the built-in example with sample processors",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a continuous background daemon",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Daemon tick interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default=None,
        help="Directory for persisting workspace state (optional)",
    )
    args = parser.parse_args()

    if args.example:
        _example()
        return

    if args.daemon:
        gw = GlobalWorkspace(state_dir=args.state_dir)
        gw.run_daemon(interval=args.interval)
        return

    # Default: run the example
    _example()


if __name__ == "__main__":
    main()
