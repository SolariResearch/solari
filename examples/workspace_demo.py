"""
Workspace Demo — Global Workspace Theory in Action

Shows how multiple processors compete for attention in a shared workspace.
No external dependencies needed — runs with just Python.

Usage:
    python examples/workspace_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solari.workspace import GlobalWorkspace, Processor, WorkspaceItem


class SensorProcessor(Processor):
    """Simulates environmental sensing."""
    name = "sensor"

    def bid(self, context):
        return [WorkspaceItem(
            source=self.name,
            content="Temperature spike to 95C detected in server room",
            item_type="threat",
            urgency=0.9,
            novelty=0.85,
            goal_relevance=0.7,
        )]


class PlannerProcessor(Processor):
    """Plans next actions based on goals."""
    name = "planner"

    def bid(self, context):
        return [WorkspaceItem(
            source=self.name,
            content="Schedule database backup for tonight",
            item_type="goal",
            urgency=0.4,
            novelty=0.3,
            goal_relevance=0.9,
        )]


class MemoryProcessor(Processor):
    """Recalls relevant past experiences."""
    name = "memory"

    def bid(self, context):
        return [WorkspaceItem(
            source=self.name,
            content="Last temperature spike (Feb 12) caused 2hr downtime",
            item_type="memory",
            urgency=0.6,
            novelty=0.5,
            goal_relevance=0.6,
        )]


class EmotionProcessor(Processor):
    """Provides emotional context."""
    name = "emotion"

    def bid(self, context):
        return [WorkspaceItem(
            source=self.name,
            content="Anxiety rising — similar pattern to previous incident",
            item_type="emotion",
            urgency=0.7,
            novelty=0.4,
            valence=-0.6,
            arousal=0.8,
        )]


def main():
    print("=" * 60)
    print("  Global Workspace Theory Demo")
    print("  4 processors competing for 7 attention slots")
    print("=" * 60)

    gw = GlobalWorkspace(capacity=7)
    gw.register_processor(SensorProcessor())
    gw.register_processor(PlannerProcessor())
    gw.register_processor(MemoryProcessor())
    gw.register_processor(EmotionProcessor())

    for tick_num in range(3):
        print(f"\n--- Tick {tick_num + 1} ---")
        result = gw.tick(context={"domain": "infrastructure"})

        ws = result.get("workspace", [])
        print(f"  Workspace items: {len(ws)}")
        for item in ws[:5]:
            src = item.get("source", "?")
            content = item.get("content", "?")[:60]
            score = item.get("score", 0)
            print(f"    [{src}] (score={score:.2f}) {content}")

        intro = result.get("introspection", {})
        pheno = result.get("phenomenal", {})
        print(f"  Coherence: {intro.get('coherence', 0):.2f}")
        print(f"  Emotion: {intro.get('meta_emotion', '?')}")
        print(f"  Felt sense: {pheno.get('felt_sense', '?')}")

    print("\n" + "=" * 60)
    print("  The sensor won — temperature spike has highest urgency.")
    print("  Memory enriched with past incident context.")
    print("  This is how coherent cognition emerges from competition.")
    print("=" * 60)


if __name__ == "__main__":
    main()
