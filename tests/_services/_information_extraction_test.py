# type: ignore
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fast_graphrag._llm._base import BaseLLMService
from fast_graphrag._models import TQueryEntities
from fast_graphrag._policies._graph_upsert import BaseGraphUpsertPolicy
from fast_graphrag._services import DefaultInformationExtractionService
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import TGraph


class TestDefaultInformationExtractionService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm_service = MagicMock(spec=BaseLLMService)
        self.llm_service.send_message = AsyncMock()
        self.chunk = MagicMock()
        self.chunk.content = "test content"
        self.chunk.id = "chunk_id"
        self.document = [self.chunk]
        self.entity_types = ["entity_type"]
        self.prompt_kwargs = {"domain": "test_domain"}
        self.service = DefaultInformationExtractionService(graph_upsert=None)
        self.service.graph_upsert = AsyncMock(spec=BaseGraphUpsertPolicy)

    @patch("fast_graphrag._services._information_extraction.format_and_send_prompt", new_callable=AsyncMock)
    async def test_extract_entities_from_query(self, mock_format_and_send_prompt):
        mock_format_and_send_prompt.return_value = (
            TQueryEntities(named=["entity1", "entity2"], generic=["generic1"]),
            None,
        )
        r = await self.service.extract_entities_from_query(self.llm_service, "test query", self.prompt_kwargs)
        named, generic = r["named"], r["generic"]
        self.assertEqual(len(named), 2)
        self.assertEqual(named[0], "ENTITY1")
        self.assertEqual(named[1], "ENTITY2")
        self.assertEqual(len(generic), 1)

    @patch("fast_graphrag._services._information_extraction.format_and_send_prompt", new_callable=AsyncMock)
    async def test_extract(self, mock_format_and_send_prompt):
        mock_format_and_send_prompt.return_value = (TGraph(entities=[], relationships=[]), [])
        tasks = self.service.extract(self.llm_service, [self.document], self.prompt_kwargs, self.entity_types)
        results = await asyncio.gather(*tasks)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], BaseGraphStorage)


if __name__ == "__main__":
    unittest.main()
