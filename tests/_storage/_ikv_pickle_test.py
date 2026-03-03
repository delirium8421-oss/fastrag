# type: ignore
import pickle
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._storage._ikv_pickle import PickleIndexedKeyValueStorage


class TestPickleIndexedKeyValueStorage(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.storage = PickleIndexedKeyValueStorage(namespace=None, config=None)
        await self.storage._insert_start()

    async def test_size(self):
        self.storage._data = {1: "value1", 2: "value2"}
        size = await self.storage.size()
        self.assertEqual(size, 2)

    async def test_get(self):
        self.storage._data = {1: "value1", 2: "value2"}
        self.storage._key_to_index = {"key1": 1, "key2": 2}
        result = await self.storage.get(["key1", "key2", "key3"])
        self.assertEqual(list(result), ["value1", "value2", None])

    async def test_get_by_index(self):
        self.storage._data = {1: "value1", 2: "value2"}
        result = await self.storage.get_by_index([1, 2, 3])
        self.assertEqual(list(result), ["value1", "value2", None])

    async def test_get_index(self):
        self.storage._key_to_index = {"key1": 1, "key2": 2}
        result = await self.storage.get_index(["key1", "key2", "key3"])
        self.assertEqual(list(result), [1, 2, None])

    async def test_upsert(self):
        await self.storage.upsert(["key1", "key2"], ["value1", "value2"])
        self.assertEqual(self.storage._data, {0: "value1", 1: "value2"})
        self.assertEqual(self.storage._key_to_index, {"key1": 0, "key2": 1})

    async def test_delete(self):
        self.storage._data = {0: "value1", 1: "value2"}
        self.storage._key_to_index = {"key1": 0, "key2": 1}
        await self.storage.delete(["key1"])
        self.assertEqual(self.storage._data, {1: "value2"})
        self.assertEqual(self.storage._key_to_index, {"key2": 1})
        self.assertEqual(self.storage._free_indices, [0])

    async def test_mask_new(self):
        self.storage._key_to_index = {"key1": 0, "key2": 1}
        result = await self.storage.mask_new([["key1", "key3"]])
        self.assertTrue(np.array_equal(result, [[False, True]]))

    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps(({0: "value"}, [1, 2, 3], {"key": 0})))
    @patch("os.path.exists", return_value=True)
    @patch("fast_graphrag._storage._ikv_pickle.logger")
    async def test_insert_start_with_existing_file(self, mock_logger, mock_exists, mock_open):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = "dummy_path"

        # Call the function
        await self.storage._insert_start()

        # Check if data was loaded correctly
        self.assertEqual(self.storage._data, {0: "value"})
        self.assertEqual(self.storage._free_indices, [1, 2, 3])
        self.assertEqual(self.storage._key_to_index, {"key": 0})
        mock_logger.debug.assert_called_once()

    @patch("fast_graphrag._storage._ikv_pickle.logger")
    async def test_insert_start_with_no_file(self, mock_logger):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = "dummy_path"

        # Call the function
        with self.assertRaises(InvalidStorageError):
            await self.storage._insert_start()

    @patch("fast_graphrag._storage._ikv_pickle.logger")
    async def test_insert_start_with_no_namespace(self, mock_logger):
        self.storage.namespace = None

        # Call the function
        await self.storage._insert_start()

        # Check if data was initialized correctly
        self.assertEqual(self.storage._data, {})
        self.assertEqual(self.storage._free_indices, [])
        mock_logger.debug.assert_called_with("Creating new volatile indexed key-value storage.")

    @patch("builtins.open", new_callable=mock_open)
    @patch("fast_graphrag._storage._ikv_pickle.logger")
    async def test_insert_done(self, mock_logger, mock_open):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_save_path.return_value = "dummy_path"
        self.storage._data = {0: "value"}
        self.storage._free_indices = [1, 2, 3]
        self.storage._key_to_index = {"key": 0}

        # Call the function
        await self.storage._insert_done()

        # Check if data was saved correctly
        mock_open.assert_called_with("dummy_path", "wb")
        mock_logger.debug.assert_called_with("Saving 1 elements to indexed key-value storage 'dummy_path'.")

    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps(({0: "value"}, [1, 2, 3], {"key": 0})))
    @patch("os.path.exists", return_value=True)
    @patch("fast_graphrag._storage._ikv_pickle.logger")
    async def test_query_start_with_existing_file(self, mock_logger, mock_exists, mock_open):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = "dummy_path"

        # Call the function
        await self.storage._query_start()

        # Check if data was loaded correctly
        self.assertEqual(self.storage._data, {0: "value"})
        self.assertEqual(self.storage._free_indices, [1, 2, 3])
        self.assertEqual(self.storage._key_to_index, {"key": 0})
        mock_logger.debug.assert_called_with("Loaded 1 elements from indexed key-value storage 'dummy_path'.")

    @patch("os.path.exists", return_value=False)
    @patch("fast_graphrag._storage._ikv_pickle.logger")
    async def test_query_start_with_no_file(self, mock_logger, mock_exists):
        self.storage.namespace = MagicMock()
        self.storage.namespace.get_load_path.return_value = None

        # Call the function
        await self.storage._query_start()

        # Check if data was initialized correctly
        self.assertEqual(self.storage._data, {})
        self.assertEqual(self.storage._free_indices, [])
        mock_logger.warning.assert_called_with(
            "No data file found for key-vector storage 'None'. Loading empty storage."
        )


if __name__ == "__main__":
    unittest.main()
