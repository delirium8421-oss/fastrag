import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fast_graphrag._utils import logger
from fast_graphrag._llm._base import BaseEmbeddingService, NoopAsyncContextManager

@dataclass
class HuggingFaceEmbeddingService(BaseEmbeddingService):
    """Embedding service using HuggingFace models."""

    embedding_dim: Optional[int] = None  # Can be set dynamically if needed
    max_token_size: int = 512
    max_elements_per_request: int = field(default=32)
    tokenizer: Any = None
    model: Any = None

    def __post_init__(self):
        self.embedding_max_requests_concurrent = (
            asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
        )
        self.embedding_per_minute_limiter = (
            AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
        )
        self.embedding_per_second_limiter = (
            AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
        )
        logger.debug("Initialized HuggingFaceEmbeddingService.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray:
        try:
            logger.debug(f"Getting embedding for texts: {texts}")

            batched_texts = [
                texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
                for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
            ]
            responses = await asyncio.gather(*[self._embedding_request(batch) for batch in batched_texts])
            embeddings = np.vstack(responses)
            logger.debug(f"Received embedding response: {len(embeddings)} embeddings")
            return embeddings
        except Exception:
            logger.exception("An error occurred during HuggingFace embedding.", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError, torch.cuda.CudaError)),
    )
    async def _embedding_request(self, input_texts: list[str]) -> np.ndarray:
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    logger.debug(f"Embedding request for batch size: {len(input_texts)}")
                    device = (
                        next(self.model.parameters()).device if torch.cuda.is_available()
                        else torch.device("mps") if torch.backends.mps.is_available()
                        else torch.device("cpu")
                    )
                    self.model = self.model.to(device)

                    encoded = self.tokenizer(
                        input_texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_token_size
                    ).to(device)

                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"]
                        )
                        embeddings = outputs.last_hidden_state.mean(dim=1)

                    if embeddings.dtype == torch.bfloat16:
                        return embeddings.detach().to(torch.float32).cpu().numpy()
                    else:
                        return embeddings.detach().cpu().numpy()
