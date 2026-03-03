# type: ignore
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

import instructor
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import RetryError

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._llm._llm_openai import OpenAIEmbeddingService, OpenAILLMService

os.environ["OPENAI_API_KEY"] = ""


RateLimitError429 = RateLimitError(message="Rate limit exceeded", response=MagicMock(), body=None)


class TestOpenAILLMService(unittest.IsolatedAsyncioTestCase):
    async def test_send_message_success(self):
        service = OpenAILLMService(api_key="test")
        mock_response = str("Hi!")
        service.llm_async_client = AsyncMock()
        service.llm_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        response, messages = await service.send_message(prompt="Hello")

        self.assertEqual(response, mock_response)
        self.assertEqual(messages[-1]["role"], "assistant")

    async def test_send_message_no_response(self):
        service = OpenAILLMService(api_key="test")
        service.llm_async_client = AsyncMock()
        service.llm_async_client.chat.completions.create.return_value = None

        with self.assertRaises(LLMServiceNoResponseError):
            await service.send_message(prompt="Hello")

    async def test_send_message_rate_limit_error(self):
        service = OpenAILLMService()
        mock_response = str("Hi!")
        async_open_ai = AsyncOpenAI(api_key="test")
        async_open_ai.chat.completions.create = AsyncMock(
            side_effect=(RateLimitError429, mock_response)
        )
        service.llm_async_client: instructor.AsyncInstructor = instructor.from_openai(
            async_open_ai
        )

        response, messages = await service.send_message(prompt="Hello", response_model=None)

        self.assertEqual(response, mock_response)
        self.assertEqual(messages[-1]["role"], "assistant")

    async def test_send_message_api_connection_error(self):
        service = OpenAILLMService()
        mock_response = str("Hi!")
        async_open_ai = AsyncOpenAI(api_key="test")
        async_open_ai.chat.completions.create = AsyncMock(
            side_effect=(APIConnectionError(request=MagicMock()), mock_response)
        )
        service.llm_async_client: instructor.AsyncInstructor = instructor.from_openai(
            async_open_ai
        )

        response, messages = await service.send_message(prompt="Hello")

        self.assertEqual(response, mock_response)
        self.assertEqual(messages[-1]["role"], "assistant")

    async def test_send_message_with_system_prompt(self):
        service = OpenAILLMService(api_key="test")
        mock_response = str("Hi!")
        service.llm_async_client = AsyncMock()
        service.llm_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        response, messages = await service.send_message(
            prompt="Hello", system_prompt="System prompt"
        )

        self.assertEqual(response, mock_response)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System prompt")

    async def test_send_message_with_history(self):
        service = OpenAILLMService(api_key="test")
        mock_response = str("Hi!")
        service.llm_async_client = AsyncMock()
        service.llm_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        history = [{"role": "user", "content": "Previous message"}]
        response, messages = await service.send_message(prompt="Hello", history_messages=history)

        self.assertEqual(response, mock_response)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Previous message")


class TestOpenAIEmbeddingService(unittest.IsolatedAsyncioTestCase):
    async def test_get_embedding_success(self):
        service = OpenAIEmbeddingService(api_key="test")
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.1, 0.2, 0.3])]
        service.embedding_async_client.embeddings.create = AsyncMock(return_value=mock_response)

        embeddings = await service.encode(texts=["test"], model="text-embedding-3-small")

        self.assertEqual(embeddings.shape, (1, 3))
        self.assertEqual(embeddings[0][0], 0.1)

    async def test_get_embedding_rate_limit_error(self):
        service = OpenAIEmbeddingService(api_key="test")
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.1, 0.2, 0.3])]
        service.embedding_async_client.embeddings.create = AsyncMock(side_effect=(RateLimitError429, mock_response))

        embeddings = await service.encode(texts=["test"], model="text-embedding-3-small")

        self.assertEqual(embeddings.shape, (1, 3))
        self.assertEqual(embeddings[0][0], 0.1)

    async def test_get_embedding_api_connection_error(self):
        service = OpenAIEmbeddingService(api_key="test")
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.1, 0.2, 0.3])]
        service.embedding_async_client.embeddings.create = AsyncMock(
            side_effect=(APIConnectionError(request=MagicMock()), mock_response)
        )
        embeddings = await service.encode(texts=["test"], model="text-embedding-3-small")

        self.assertEqual(embeddings.shape, (1, 3))
        self.assertEqual(embeddings[0][0], 0.1)

    async def test_get_embedding_retry_failure(self):
        service = OpenAIEmbeddingService(api_key="test")
        service.embedding_async_client.embeddings.create = AsyncMock(
            side_effect=RateLimitError429
        )

        with self.assertRaises(RetryError):
            await service.encode(texts=["test"], model="text-embedding-3-small")

    async def test_get_embedding_with_different_model(self):
        service = OpenAIEmbeddingService(api_key="test")
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.4, 0.5, 0.6])]
        service.embedding_async_client.embeddings.create = AsyncMock(return_value=mock_response)

        embeddings = await service.encode(texts=["test"], model="text-embedding-3-large")

        self.assertEqual(embeddings.shape, (1, 3))
        self.assertEqual(embeddings[0][0], 0.4)


if __name__ == "__main__":
    unittest.main()
