"""Basic tests for Solari toolkit."""
import unittest
import tempfile
import os


class TestImports(unittest.TestCase):
    def test_import_solari(self):
        import solari
        self.assertTrue(hasattr(solari, "__version__"))

    def test_import_workspace(self):
        from solari.workspace import GlobalWorkspace, Processor, WorkspaceItem
        gw = GlobalWorkspace(capacity=5)
        self.assertEqual(gw.capacity, 5)

    def test_import_query(self):
        from solari.query import list_minds
        self.assertIsNotNone(list_minds)

    def test_import_ingest(self):
        from solari.ingest import main
        self.assertIsNotNone(main)

    def test_import_cli(self):
        from solari.cli import main
        self.assertIsNotNone(main)


class TestWorkspace(unittest.TestCase):
    def test_create_workspace(self):
        from solari.workspace import GlobalWorkspace
        gw = GlobalWorkspace(capacity=7)
        self.assertEqual(gw.capacity, 7)

    def test_tick_empty(self):
        from solari.workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        result = gw.tick()
        self.assertIn("workspace", result)

    def test_register_processor(self):
        from solari.workspace import GlobalWorkspace, Processor, WorkspaceItem
        class TestProc(Processor):
            name = "test"
            def bid(self, context):
                return [WorkspaceItem(source="test", content="hello", item_type="test")]
        gw = GlobalWorkspace()
        gw.register_processor(TestProc())
        result = gw.tick()
        self.assertTrue(len(result.get("workspace", [])) > 0)


class TestQuery(unittest.TestCase):
    def test_list_empty_dir(self):
        from solari.query import list_minds
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            result = list_minds(Path(td))
            self.assertEqual(len(result), 0)


class TestConfidencePruning(unittest.TestCase):
    def test_ingest_stores_confidence(self):
        """New entries should store their confidence value."""
        from solari.ingest import ingest_into_mind
        import gzip
        import json

        with tempfile.TemporaryDirectory() as td:
            n = ingest_into_mind(
                "test_mind",
                ["This is a test entry about machine learning algorithms."],
                source="test",
                minds_dir=td,
                confidence=0.8,
            )
            self.assertGreater(n, 0)
            meta_path = os.path.join(td, "test_mind", "metadata.json.gz")
            with gzip.open(meta_path, "rt") as f:
                meta = json.load(f)
            self.assertTrue(all(e.get("confidence") == 0.8 for e in meta))

    def test_higher_confidence_supersedes(self):
        """A higher-confidence entry should replace a similar lower-confidence one."""
        from solari.ingest import ingest_into_mind
        import gzip
        import json

        with tempfile.TemporaryDirectory() as td:
            # Ingest at low confidence
            n1 = ingest_into_mind(
                "test_mind",
                ["Machine learning algorithms optimize a loss function."],
                source="old_source",
                minds_dir=td,
                confidence=0.3,
            )
            meta_path = os.path.join(td, "test_mind", "metadata.json.gz")
            with gzip.open(meta_path, "rt") as f:
                before = json.load(f)
            count_before = len(before)

            # Ingest very similar content at higher confidence
            n2 = ingest_into_mind(
                "test_mind",
                ["Machine learning algorithms optimize a loss function to minimize error."],
                source="new_source",
                minds_dir=td,
                confidence=0.9,
            )
            with gzip.open(meta_path, "rt") as f:
                after = json.load(f)

            # The old low-confidence entry should have been pruned
            old_entries = [e for e in after if e.get("source") == "old_source"]
            new_entries = [e for e in after if e.get("source") == "new_source"]
            self.assertEqual(len(old_entries), 0, "Old entry should be superseded")
            self.assertGreater(len(new_entries), 0, "New entry should exist")

    def test_lower_confidence_does_not_supersede(self):
        """A lower-confidence entry should NOT replace a higher-confidence one."""
        from solari.ingest import ingest_into_mind
        import gzip
        import json

        with tempfile.TemporaryDirectory() as td:
            # Ingest at high confidence
            ingest_into_mind(
                "test_mind",
                ["Neural networks use backpropagation for training."],
                source="high_source",
                minds_dir=td,
                confidence=0.9,
            )
            meta_path = os.path.join(td, "test_mind", "metadata.json.gz")
            with gzip.open(meta_path, "rt") as f:
                before = json.load(f)

            # Ingest similar at lower confidence
            ingest_into_mind(
                "test_mind",
                ["Neural networks use backpropagation for training weights."],
                source="low_source",
                minds_dir=td,
                confidence=0.3,
            )
            with gzip.open(meta_path, "rt") as f:
                after = json.load(f)

            # High-confidence entry should still be there
            high = [e for e in after if e.get("source") == "high_source"]
            self.assertGreater(len(high), 0, "High-confidence entry should remain")


if __name__ == "__main__":
    unittest.main()
