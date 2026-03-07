"""Ollama LLM and embedding services for fast-graphrag."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type

import aiohttp
import numpy as np
from aiolimiter import AsyncLimiter
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fast_graphrag._utils import logger
from fast_graphrag._llm._base import (
    BaseEmbeddingService,
    BaseLLMService,
    NoopAsyncContextManager,
    T_model,
)
from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._types import BaseModelAlias


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON from text, handling markdown code blocks and wrapped content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dict if found, None otherwise
    """
    text = text.strip()
    
    # Try to extract JSON from markdown code blocks
    # Pattern: ```json\n{...}\n``` or ```\n{...}\n```
    json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        try:
            json_str = match.group(1).strip()
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try direct JSON parsing (handles both wrapped and unwrapped)
    # First try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in the text
    # Look for the first { and find its matching }
    start_idx = text.find('{')
    if start_idx >= 0:
        # Count braces to find matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        json_str = text[start_idx:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                    break
    
    return None


def _auto_fix_graph_json(json_data: dict) -> dict:
    """Auto-fix incomplete Graph JSON by filling in missing required fields.

    Args:
        json_data: Potentially incomplete JSON data

    Returns:
        Fixed JSON data with all required fields
    """
    # Ensure top-level structure exists
    if "entities" not in json_data:
        json_data["entities"] = []
    if "relationships" not in json_data:
        json_data["relationships"] = []

    # Fix entities
    if isinstance(json_data["entities"], list):
        for i, entity in enumerate(json_data["entities"]):
            if isinstance(entity, dict):
                if "name" not in entity:
                    entity["name"] = f"UNKNOWN_ENTITY_{i}"
                if "type" not in entity:
                    entity["type"] = "UNKNOWN"
                if "desc" not in entity or not entity["desc"]:
                    entity["desc"] = f"Entity {entity.get('name', 'unknown')}"

    # Fix relationships
    if isinstance(json_data["relationships"], list):
        fixed_relationships = []
        for i, rel in enumerate(json_data["relationships"]):
            if isinstance(rel, dict):
                # Only keep relationships with both source and target
                has_source = "source" in rel and rel["source"]
                has_target = "target" in rel and rel["target"]

                if has_source and has_target:
                    # Fix description if missing
                    if "desc" not in rel or not rel["desc"]:
                        rel["desc"] = f"Relationship between {rel['source']} and {rel['target']}"
                    fixed_relationships.append(rel)
                else:
                    # Skip incomplete relationships that can't be salvaged
                    logger.warning(f"Skipping incomplete relationship at index {i}: missing {'source' if not has_source else 'target'}")

        json_data["relationships"] = fixed_relationships

    return json_data


def _validate_graph_json(json_data: dict) -> list[str]:
    """Validate extracted JSON for Graph structure.

    Args:
        json_data: Extracted JSON data

    Returns:
        List of validation errors, empty list if valid
    """
    errors = []

    # Check top-level structure
    if "entities" not in json_data:
        errors.append("Missing 'entities' field")
    if "relationships" not in json_data:
        errors.append("Missing 'relationships' field")
    
    # Validate entities
    if "entities" in json_data:
        if not isinstance(json_data["entities"], list):
            errors.append("'entities' must be a list")
        else:
            for i, entity in enumerate(json_data["entities"]):
                if not isinstance(entity, dict):
                    errors.append(f"entities[{i}] must be a dict, got {type(entity).__name__}")
                    continue
                
                # Check required fields
                if "name" not in entity:
                    errors.append(f"entities[{i}] missing 'name' field")
                if "type" not in entity:
                    errors.append(f"entities[{i}] missing 'type' field")
                if "desc" not in entity or not entity["desc"]:
                    entity_name = entity.get("name", f"entity[{i}]")
                    errors.append(f"entities[{i}] ('{entity_name}') missing 'desc' field or empty description")
    
    # Validate relationships
    if "relationships" in json_data:
        if not isinstance(json_data["relationships"], list):
            errors.append("'relationships' must be a list")
        else:
            for i, rel in enumerate(json_data["relationships"]):
                if not isinstance(rel, dict):
                    errors.append(f"relationships[{i}] must be a dict")
                    continue
                
                if "source" not in rel:
                    errors.append(f"relationships[{i}] missing 'source' field")
                if "target" not in rel:
                    errors.append(f"relationships[{i}] missing 'target' field")
                if "desc" not in rel or not rel["desc"]:
                    errors.append(f"relationships[{i}] missing 'desc' field or empty description")
    
    return errors

@dataclass
class OllamaLLMService(BaseLLMService):
    """LLM Service for Ollama models."""

    model: str = field(default="qwen2.5-14b-instruct")
    timeout: int = field(default=1800)  # Increased to 30 minutes for slow models
    request_timeout: int = field(default=600)  # Per-request timeout (10 minutes)

    def __post_init__(self):
        self.llm_max_requests_concurrent = (
            asyncio.Semaphore(self.max_requests_concurrent)
            if self.rate_limit_concurrency
            else NoopAsyncContextManager()
        )
        self.llm_per_minute_limiter = (
            AsyncLimiter(self.max_requests_per_minute, 60)
            if self.rate_limit_per_minute
            else NoopAsyncContextManager()
        )
        self.llm_per_second_limiter = (
            AsyncLimiter(self.max_requests_per_second, 1)
            if self.rate_limit_per_second
            else NoopAsyncContextManager()
        )
        self.session = None
        logger.debug("Initialized OllamaLLMService.")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
            )
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self.session

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def send_message(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[T_model] | None = None,
        **kwargs: Any,
    ) -> Tuple[T_model, list[dict[str, str]]]:
        """Send a message to Ollama and receive a response.

        Args:
            prompt: The input message to send
            system_prompt: The system prompt for context
            history_messages: Previous messages in conversation
            response_model: Pydantic model to parse response
            **kwargs: Additional arguments

        Returns:
            Tuple of (response_parsed, messages_list)
        """
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=4, max=30),
            retry=retry_if_exception_type((
                aiohttp.ClientConnectorError,
                aiohttp.ClientError,
                asyncio.TimeoutError,
                LLMServiceNoResponseError
            )),
        )
        async def _send_request():
            async with self.llm_max_requests_concurrent:
                async with self.llm_per_minute_limiter:
                    async with self.llm_per_second_limiter:
                        try:
                            logger.debug(f"Sending message to Ollama: {prompt[:100]}...")
                            session = await self._get_session()

                            # Build messages list
                            messages: list[dict[str, str]] = []

                            if system_prompt:
                                messages.append({"role": "system", "content": system_prompt})
                                logger.debug(f"Added system prompt")

                            if history_messages:
                                messages.extend(history_messages)
                                logger.debug(f"Added {len(history_messages)} history messages")

                            messages.append({"role": "user", "content": prompt})

                            # Normalize URL - convert https:// to http:// for localhost
                            base_url = self.base_url
                            if ("localhost" in base_url or "127.0.0.1" in base_url) and base_url.startswith("https://"):
                                base_url = base_url.replace("https://", "http://")
                                logger.debug(f"Normalized localhost URL from https to http: {base_url}")

                            # Prepare Ollama API call
                            url = f"{base_url}/api/chat"
                            payload = {
                                "model": self.model,
                                "messages": messages,
                                "stream": False,
                                "options": {
                                    "temperature": kwargs.get("temperature", 0.0),
                                    "num_ctx": kwargs.get("num_ctx", 32768),
                                    "top_p": kwargs.get("top_p", 1),
                                },
                            }
                            
                            # Request JSON format if response_model is provided
                            if response_model:
                                payload["format"] = "json"

                            # Configure timeout for this request
                            timeout = aiohttp.ClientTimeout(total=self.request_timeout)

                            # Call Ollama API
                            async with session.post(url, json=payload, timeout=timeout) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    response_text = result.get("message", {}).get("content", "")

                                    if not response_text:
                                        logger.error("No response text received from Ollama")
                                        raise LLMServiceNoResponseError(
                                            "No response received from Ollama"
                                        )

                                    # Parse response if response_model is provided
                                    if response_model:
                                        try:
                                            # Try to extract and parse JSON
                                            json_data = _extract_json_from_text(response_text)
                                            if json_data is None:
                                                raise json.JSONDecodeError(
                                                    f"Could not extract JSON from response: {response_text[:200]}",
                                                    response_text,
                                                    0
                                                )
                                            
                                            # Validate if it's a Graph type
                                            model_name = response_model.__name__ if hasattr(response_model, '__name__') else str(response_model)
                                            
                                            if model_name == "TGraph" or "Graph" in model_name:
                                                validation_errors = _validate_graph_json(json_data)
                                                if validation_errors:
                                                    # Apply auto-fix to salvage incomplete data
                                                    logger.warning(f"Validation errors detected: {', '.join(validation_errors)}")
                                                    logger.info("Applying auto-fix to repair incomplete Graph JSON")
                                                    json_data = _auto_fix_graph_json(json_data)

                                                    # Re-validate after fix
                                                    revalidation_errors = _validate_graph_json(json_data)
                                                    if revalidation_errors:
                                                        # Still invalid after auto-fix, trigger retry
                                                        error_msg = "JSON validation failed after auto-fix:\n" + "\n".join(revalidation_errors)
                                                        logger.error(f"Auto-fix failed: {error_msg}\nResponse: {response_text[:300]}")
                                                        raise ValueError(error_msg)
                                                    else:
                                                        logger.info("✓ Auto-fix successful, proceeding with repaired data")
                                            
                                            # DEBUG: Log the actual JSON data before validation (only for debugging)
                                            logger.debug(f"🔍 DEBUG - Model: {model_name}")
                                            logger.debug(f"🔍 DEBUG - JSON Data: {json.dumps(json_data, indent=2)}")

                                            # Handle BaseModelAlias types which have a .Model inner class
                                            if issubclass(response_model, BaseModelAlias):
                                                # Use the Pydantic Model class to parse
                                                pydantic_model = response_model.Model(**json_data)
                                                # Convert to dataclass using to_dataclass
                                                llm_response = pydantic_model.to_dataclass(pydantic_model)
                                            else:
                                                # Regular Pydantic model
                                                llm_response = response_model(**json_data)
                                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                                            logger.error(
                                                f"Failed to parse response as {response_model.__name__}: {e}. "
                                                f"Response text: {response_text[:300]}"
                                            )
                                            raise LLMServiceNoResponseError(
                                                f"Failed to parse response as {response_model.__name__}: {str(e)}"
                                            )
                                    else:
                                        llm_response = response_text

                                    # Add response to messages
                                    messages.append(
                                        {
                                            "role": "assistant",
                                            "content": (
                                                llm_response.model_dump_json()
                                                if isinstance(llm_response, BaseModel)
                                                else str(response_text)
                                            ),
                                        }
                                    )

                                    logger.debug(
                                        f"Received response from Ollama: {str(llm_response)[:100]}..."
                                    )
                                    return llm_response, messages
                                else:
                                    error_text = await response.text()
                                    raise RuntimeError(
                                        f"Ollama API error (status {response.status}): {error_text}"
                                    )

                        except Exception:
                            logger.exception("Error sending message to Ollama", exc_info=True)
                            raise
        
        return await _send_request()


@dataclass
class OllamaEmbeddingService(BaseEmbeddingService):
    """Embedding service using Ollama."""

    embedding_dim: int = field(default=1024)
    timeout: int = field(default=600)
    max_elements_per_request: int = field(default=32)

    def __post_init__(self):
        self.embedding_max_requests_concurrent = (
            asyncio.Semaphore(self.max_requests_concurrent)
            if self.rate_limit_concurrency
            else NoopAsyncContextManager()
        )
        self.embedding_per_minute_limiter = (
            AsyncLimiter(self.max_requests_per_minute, 60)
            if self.rate_limit_per_minute
            else NoopAsyncContextManager()
        )
        self.embedding_per_second_limiter = (
            AsyncLimiter(self.max_requests_per_second, 1)
            if self.rate_limit_per_second
            else NoopAsyncContextManager()
        )
        self.session = None
        logger.debug("Initialized OllamaEmbeddingService.")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
            )
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self.session

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def encode(
        self, texts: list[str], model: Optional[str] = None
    ) -> np.ndarray:
        """Get embeddings for texts from Ollama.

        Args:
            texts: List of text strings to embed
            model: Optional model override

        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        try:
            logger.debug(f"Getting embeddings for {len(texts)} texts")

            # Use provided model or fall back to self.model
            embedding_model = model or self.model
            if not embedding_model:
                raise ValueError(
                    "No model specified for Ollama embedding. Set via 'model' parameter or attribute."
                )

            # Batch texts
            batched_texts = [
                texts[
                    i
                    * self.max_elements_per_request : (i + 1)
                    * self.max_elements_per_request
                ]
                for i in range(
                    (len(texts) + self.max_elements_per_request - 1)
                    // self.max_elements_per_request
                )
            ]

            responses = await asyncio.gather(
                *[self._embedding_request(batch, embedding_model) for batch in batched_texts]
            )
            embeddings = np.vstack(responses)
            logger.debug(f"Received {len(embeddings)} embeddings")
            return embeddings
        except Exception:
            logger.exception("An error occurred during Ollama embedding.", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError, aiohttp.ClientError)),
    )
    async def _embedding_request(
        self, input_texts: list[str], model: str
    ) -> np.ndarray:
        """Make embedding request to Ollama API.

        Args:
            input_texts: Texts to embed
            model: Model name to use

        Returns:
            NumPy array of embeddings with shape (len(input_texts), embedding_dim)
        """
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    logger.debug(f"Embedding request for batch size: {len(input_texts)}")
                    session = await self._get_session()

                    # Normalize URL - convert https:// to http:// for localhost
                    base_url = self.base_url
                    if ("localhost" in base_url or "127.0.0.1" in base_url) and base_url.startswith("https://"):
                        base_url = base_url.replace("https://", "http://")
                        logger.debug(f"Normalized localhost URL from https to http: {base_url}")

                    url = f"{base_url}/api/embed"
                    embeddings_list = []
                    
                    for text in input_texts:
                        try:
                            payload = {"model": model, "input": text}
                            async with session.post(url, json=payload) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    embedding = np.array(result.get("embedding", []))
                                    if len(embedding) == 0:
                                        logger.warning(
                                            f"Empty embedding returned for text: {text[:50]}..."
                                        )
                                        # Return zero embedding if empty
                                        embedding = np.zeros(self.embedding_dim)
                                    embeddings_list.append(embedding)
                                else:
                                    error_text = await response.text()
                                    raise RuntimeError(
                                        f"Ollama embedding failed with status {response.status}: {error_text}"
                                    )
                        except aiohttp.ClientError as e:
                            logger.error(f"Error embedding text: {e}")
                            raise
                    
                    # Stack all embeddings
                    if embeddings_list:
                        return np.vstack(embeddings_list)
                    else:
                        return np.zeros((len(input_texts), self.embedding_dim))
