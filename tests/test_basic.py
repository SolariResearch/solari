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


if __name__ == "__main__":
    unittest.main()
