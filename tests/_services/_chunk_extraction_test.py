# type: ignore
import unittest
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import patch

import xxhash

from fast_graphrag._services._chunk_extraction import DefaultChunkingService
from fast_graphrag._types import THash


@dataclass
class MockDocument:
    data: str
    metadata: Dict[str, Any]


@dataclass
class MockChunk:
    id: THash
    content: str
    metadata: Dict[str, Any]


class TestDefaultChunkingService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chunking_service = DefaultChunkingService()

    async def test_extract(self):
        doc1 = MockDocument(data="test data 1", metadata={"meta": "data1"})
        doc2 = MockDocument(data="test data 2", metadata={"meta": "data2"})
        documents = [doc1, doc2]

        with patch.object(
            self.chunking_service,
            "_extract_chunks",
            return_value=[
                MockChunk(id=THash(xxhash.xxh3_64_intdigest(doc1.data) // 2), content=doc1.data, metadata=doc1.metadata)
            ],
        ) as mock_extract_chunks:
            chunks = await self.chunking_service.extract(documents)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 1)
        self.assertEqual(chunks[0][0].content, "test data 1")
        self.assertEqual(chunks[0][0].metadata, {"meta": "data1"})
        mock_extract_chunks.assert_called()

    async def test_extract_with_duplicates(self):
        doc1 = MockDocument(data="test data 1", metadata={"meta": "data1"})
        doc2 = MockDocument(data="test data 1", metadata={"meta": "data1"})
        documents = [doc1, doc2]

        with patch.object(
            self.chunking_service,
            "_extract_chunks",
            return_value=[
                MockChunk(id=THash(xxhash.xxh3_64_intdigest(doc1.data) // 2), content=doc1.data, metadata=doc1.metadata)
            ],
        ) as mock_extract_chunks:
            chunks = await self.chunking_service.extract(documents)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 1)
        self.assertEqual(len(chunks[1]), 1)
        self.assertEqual(chunks[0][0].content, "test data 1")
        self.assertEqual(chunks[0][0].metadata, {"meta": "data1"})
        self.assertEqual(chunks[1][0].content, "test data 1")
        self.assertEqual(chunks[1][0].metadata, {"meta": "data1"})
        mock_extract_chunks.assert_called()

    async def test_extract_chunks(self):
        doc = MockDocument(data="test data", metadata={"meta": "data"})
        chunk = MockChunk(id=THash(xxhash.xxh3_64_intdigest(doc.data) // 2), content=doc.data, metadata=doc.metadata)

        chunks = await self.chunking_service._extract_chunks(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].id, chunk.id)
        self.assertEqual(chunks[0].content, chunk.content)
        self.assertEqual(chunks[0].metadata, chunk.metadata)


if __name__ == "__main__":
    unittest.main()
