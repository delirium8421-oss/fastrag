# type: ignore
import unittest
from unittest.mock import AsyncMock, patch

from fast_graphrag._storage._base import BaseStorage


class TestBaseStorage(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.storage = BaseStorage(config=None)

    @patch.object(BaseStorage, '_insert_start', new_callable=AsyncMock)
    @patch.object(BaseStorage, '_query_done', new_callable=AsyncMock)
    @patch("fast_graphrag._storage._base.logger")
    async def test_insert_start_from_query_mode(self, mock_logger, mock_query_done, mock_insert_start):
        self.storage._mode = "query"
        self.storage._in_progress = True

        await self.storage.insert_start()

        mock_query_done.assert_called_once()
        mock_insert_start.assert_called_once()
        mock_logger.error.assert_called_once()
        self.assertEqual(self.storage._mode, "insert")
        self.assertFalse(self.storage._in_progress)

    @patch.object(BaseStorage, '_insert_start', new_callable=AsyncMock)
    async def test_insert_start_from_none_mode(self, mock_insert_start):
        self.storage._mode = None
        self.storage._in_progress = False

        await self.storage.insert_start()

        mock_insert_start.assert_called_once()
        self.assertEqual(self.storage._mode, "insert")
        self.assertFalse(self.storage._in_progress)

    @patch.object(BaseStorage, '_insert_done', new_callable=AsyncMock)
    async def test_insert_done_in_insert_mode(self, mock_insert_done):
        self.storage._mode = "insert"
        self.storage._in_progress = True

        await self.storage.insert_done()

        mock_insert_done.assert_called_once()

    @patch("fast_graphrag._storage._base.logger")
    async def test_insert_done_in_query_mode(self, mock_logger):
        self.storage._mode = "query"
        self.storage._in_progress = True

        await self.storage.insert_done()
        mock_logger.error.assert_called_once()

    @patch.object(BaseStorage, '_query_start', new_callable=AsyncMock)
    @patch.object(BaseStorage, '_insert_done', new_callable=AsyncMock)
    @patch("fast_graphrag._storage._base.logger")
    async def test_query_start_from_insert_mode(self, mock_logger, mock_insert_done, mock_query_start):
        self.storage._mode = "insert"
        self.storage._in_progress = True

        await self.storage.query_start()

        mock_insert_done.assert_called_once()
        mock_query_start.assert_called_once()
        mock_logger.error.assert_called_once()
        self.assertEqual(self.storage._mode, "query")
        self.assertFalse(self.storage._in_progress)

    @patch.object(BaseStorage, '_query_start', new_callable=AsyncMock)
    async def test_query_start_from_none_mode(self, mock_query_start):
        self.storage._mode = None
        self.storage._in_progress = False

        await self.storage.query_start()

        mock_query_start.assert_called_once()
        self.assertEqual(self.storage._mode, "query")

    @patch.object(BaseStorage, '_query_done', new_callable=AsyncMock)
    async def test_query_done_in_query_mode(self, mock_query_done):
        self.storage._mode = "query"
        self.storage._in_progress = True

        await self.storage.query_done()

        mock_query_done.assert_called_once()

    @patch("fast_graphrag._storage._base.logger")
    async def test_query_done_in_insert_mode(self, mock_logger):
        self.storage._mode = "insert"
        self.storage._in_progress = True

        await self.storage.query_done()
        mock_logger.error.assert_called_once()

if __name__ == "__main__":
    unittest.main()
