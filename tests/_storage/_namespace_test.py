import gc
import os
import shutil
import unittest
from typing import cast

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._storage._namespace import Namespace, Workspace


class TestWorkspace(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        def _(self: Workspace) -> None:
            pass

        Workspace.__del__ = _
        self.test_dir = "test_workspace"
        self.workspace = Workspace(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_new_workspace(self):
        ws = Workspace.new(self.test_dir)
        self.assertIsInstance(ws, Workspace)
        self.assertEqual(ws.working_dir, self.test_dir)

    def test_get_load_path_no_checkpoint(self):
        self.assertEqual(self.workspace.get_load_path(), None)

    def test_get_save_path_creates_directory(self):
        save_path = self.workspace.get_save_path()
        self.assertTrue(os.path.exists(save_path))

    async def test_with_checkpoint_failures(self):
        for checkpoint in [1, 2, 3]:
            os.makedirs(os.path.join(self.test_dir, str(checkpoint)))
        self.workspace = Workspace(self.test_dir)

        async def sample_fn():
            if "1" not in cast(str, self.workspace.get_load_path()):
                raise Exception("Checkpoint not loaded")
            return "success"

        result = await self.workspace.with_checkpoints(sample_fn)
        self.assertEqual(result, "success")
        self.assertEqual(self.workspace.current_load_checkpoint, 1)
        self.assertEqual(self.workspace.failed_checkpoints, ["3", "2"])

    async def test_with_checkpoint_no_failure(self):
        for checkpoint in [1, 2, 3]:
            os.makedirs(os.path.join(self.test_dir, str(checkpoint)))
        self.workspace = Workspace(self.test_dir)

        async def sample_fn():
            return "success"

        result = await self.workspace.with_checkpoints(sample_fn)
        self.assertEqual(result, "success")
        self.assertEqual(self.workspace.current_load_checkpoint, 3)
        self.assertEqual(self.workspace.failed_checkpoints, [])

    async def test_with_checkpoint_all_failures(self):
        for checkpoint in [1, 2, 3]:
            os.makedirs(os.path.join(self.test_dir, str(checkpoint)))
        self.workspace = Workspace(self.test_dir)

        async def sample_fn():
            raise Exception("Checkpoint not loaded")

        with self.assertRaises(InvalidStorageError):
            await self.workspace.with_checkpoints(sample_fn)
        self.assertEqual(self.workspace.current_load_checkpoint, None)
        self.assertEqual(self.workspace.failed_checkpoints, ["3", "2", "1"])

    async def test_with_checkpoint_all_failures_accept_none(self):
        for checkpoint in [1, 2, 3]:
            os.makedirs(os.path.join(self.test_dir, str(checkpoint)))
        self.workspace = Workspace(self.test_dir)

        async def sample_fn():
            if self.workspace.get_load_path() is not None:
                raise Exception("Checkpoint not loaded")

        result = await self.workspace.with_checkpoints(sample_fn)
        self.assertEqual(result, None)
        self.assertEqual(self.workspace.current_load_checkpoint, None)
        self.assertEqual(self.workspace.failed_checkpoints, ["3", "2", "1"])


class TestNamespace(unittest.TestCase):
    def setUp(self):
        def _(self: Workspace) -> None:
            pass

        Workspace.__del__ = _

    def test_get_load_path_no_checkpoint_no_file(self):
        self.test_dir = "test_workspace"
        self.workspace = Workspace(self.test_dir)
        self.workspace.__del__ = lambda: None
        self.namespace = Namespace(self.workspace, "test_namespace")
        self.assertEqual(None, self.namespace.get_load_path("resource"))
        del self.workspace
        gc.collect()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_load_path_no_checkpoint_with_file(self):
        self.test_dir = "test_workspace"
        self.workspace = Workspace(self.test_dir)
        self.workspace.__del__ = lambda: None
        self.namespace = Namespace(self.workspace, "test_namespace")

        with open(os.path.join(self.test_dir, "test_namespace_resource"), "w") as f:
            f.write("test")
        self.assertEqual(
            os.path.join("test_workspace", "test_namespace_resource"), self.namespace.get_load_path("resource")
        )
        del self.workspace
        gc.collect()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_load_path_with_checkpoint(self):
        self.test_dir = "test_workspace"
        self.workspace = Workspace(self.test_dir)
        self.namespace = Namespace(self.workspace, "test_namespace")
        self.workspace.current_load_checkpoint = 1
        load_path = self.namespace.get_load_path("resource")
        self.assertEqual(load_path, os.path.join(self.test_dir, "1", "test_namespace_resource"))

        del self.workspace
        gc.collect()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_save_path_creates_directory(self):
        self.test_dir = "test_workspace"
        self.workspace = Workspace(self.test_dir)
        self.namespace = Namespace(self.workspace, "test_namespace")
        save_path = self.namespace.get_save_path("resource")
        self.assertTrue(os.path.exists(os.path.join(*os.path.split(save_path)[:-1])))

        del self.workspace
        gc.collect()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
