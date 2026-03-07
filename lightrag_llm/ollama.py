from collections.abc import AsyncIterator
import os
import re
import asyncio

import pipmaster as pm

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")

import ollama
from ollama._types import ResponseError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from lightrag.api import __api_version__

import numpy as np
from typing import Optional, Union
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    logger,
)


_OLLAMA_CLOUD_HOST = "https://ollama.com"
_CLOUD_MODEL_SUFFIX_PATTERN = re.compile(r"(?:-cloud|:cloud)$")


def _coerce_host_for_cloud_model(host: Optional[str], model: object) -> Optional[str]:
    if host:
        return host
    try:
        model_name_str = str(model) if model is not None else ""
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to convert model to string: {e}, using empty string")
        model_name_str = ""
    if _CLOUD_MODEL_SUFFIX_PATTERN.search(model_name_str):
        logger.debug(
            f"Detected cloud model '{model_name_str}', using Ollama Cloud host"
        )
        return _OLLAMA_CLOUD_HOST
    return host


def _should_retry_on_response_error(exc: Exception) -> bool:
    """Determine if we should retry based on ResponseError status code."""
    if isinstance(exc, ResponseError):
        # Retry on 5xx server errors (500, 503, etc.)
        if hasattr(exc, 'status_code') and exc.status_code and exc.status_code >= 500:
            return True
        # Retry on connection timeouts (status_code -1)
        if hasattr(exc, 'status_code') and exc.status_code == -1:
            return True
    return False


# @retry(
#     stop=stop_after_attempt(5),
#     wait=wait_exponential(multiplier=2, min=4, max=30),
#     retry=retry_if_exception_type(
#         (RateLimitError, APIConnectionError, APITimeoutError, ResponseError)
#     ),
# )
# async def _ollama_model_if_cache(
#     model,
#     prompt,
#     system_prompt=None,
#     history_messages=[],
#     enable_cot: bool = False,
#     **kwargs,
# ) -> Union[str, AsyncIterator[str]]:
#     if enable_cot:
#         logger.debug("enable_cot=True is not supported for ollama and will be ignored.")
#     stream = True if kwargs.get("stream") else False

#     kwargs.pop("max_tokens", None)
#     # kwargs.pop("response_format", None) # allow json
#     host = kwargs.pop("host", None)
#     timeout = kwargs.pop("timeout", None)
#     if timeout == 0:
#         timeout = None
#     kwargs.pop("hashing_kv", None)
#     api_key = kwargs.pop("api_key", None)
#     # fallback to environment variable when not provided explicitly
#     if not api_key:
#         api_key = os.getenv("OLLAMA_API_KEY")
#     headers = {
#         "Content-Type": "application/json",
#         "User-Agent": f"LightRAG/{__api_version__}",
#     }
#     if api_key:
#         headers["Authorization"] = f"Bearer {api_key}"

#     host = _coerce_host_for_cloud_model(host, model)

#     # Increase timeout for Ollama to handle slower models
#     if timeout is None:
#         timeout = 300  # 5 minutes default
    
#     ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

#     try:
#         print("OLLAMA CHAT PAYLOAD")
#         print("model:", model)
#         print("kwargs:", kwargs)

#         messages = []
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})
#         #print("HISTORY MESSAGES : ", history_messages, "INSIDE OLLAMA OLLAMA MODEL IF CACHE")
#         messages.extend(history_messages)
#         messages.append({"role": "user", "content": prompt})
#         print("messages : ", messages)
#         kwargs.pop("options", None)
#         response = await ollama_client.chat(model=model, messages=messages, **kwargs)
#         if stream:
#             """cannot cache stream response and process reasoning"""

#             async def inner():
#                 try:
#                     async for chunk in response:
#                         yield chunk["message"]["content"]
#                 except Exception as e:
#                     logger.error(f"Error in stream response: {str(e)}")
#                     raise
#                 finally:
#                     try:
#                         await ollama_client._client.aclose()
#                         logger.debug("Successfully closed Ollama client for streaming")
#                     except Exception as close_error:
#                         logger.warning(f"Failed to close Ollama client: {close_error}")

#             return inner()
#         else:
#             model_response = response["message"]["content"]

#             """
#             If the model also wraps its thoughts in a specific tag,
#             this information is not needed for the final
#             response and can simply be trimmed.
#             """

#             return model_response
#     except ResponseError as e:
#         print("OLLAMA CHAT PAYLOAD")
#         print("model:", model)
#         print("messages:", messages)
#         print("kwargs:", kwargs)

#         try:
#             await ollama_client._client.aclose()
#             logger.debug("Successfully closed Ollama client after ResponseError")
#         except Exception as close_error:
#             logger.warning(
#                 f"Failed to close Ollama client after ResponseError: {close_error}"
#             )
#         # Log server error details for debugging
#         import traceback
#         traceback.print_exc()
#         logger.error(
#             f"Ollama server error (status {getattr(e, 'status_code', 'unknown')}): {str(e)}"
#         )
#         raise e
#     except Exception as e:
#         try:
#             await ollama_client._client.aclose()
#             logger.debug("Successfully closed Ollama client after exception")
#         except Exception as close_error:
#             logger.warning(
#                 f"Failed to close Ollama client after exception: {close_error}"
#             )
#         raise e
#     finally:
#         if not stream:
#             try:
#                 await ollama_client._client.aclose()
#                 logger.debug(
#                     "Successfully closed Ollama client for non-streaming response"
#                 )
#             except Exception as close_error:
#                 logger.warning(
#                     f"Failed to close Ollama client in finally block: {close_error}"
#                 )


# async def ollama_model_complete(
#     prompt,
#     system_prompt=None,
#     history_messages=[],
#     enable_cot: bool = False,
#     keyword_extraction=False,
#     **kwargs,
# ) -> Union[str, AsyncIterator[str]]:
#     if not history_messages:
#         history_messages = []
#     #print("INSIDE OLLAMA MODEL COMPLETE FUNCTION : ", "HISTORY MESSGES ", history_messages)
#     keyword_extraction = kwargs.pop("keyword_extraction", None)
#     if keyword_extraction:
#         kwargs["format"] = "json"
#     model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
#     return await _ollama_model_if_cache(
#         model_name,
#         prompt,
#         system_prompt=system_prompt,
#         history_messages=history_messages,
#         enable_cot=enable_cot,
#         **kwargs,
#     )

async def collect_async_generator(gen):
    parts = []
    async for chunk in gen:
        if chunk:
            parts.append(chunk)
    return "".join(parts)

# FIX FOR 500 ISSUE: 
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, ResponseError)
    ),
)
async def _ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:

    if history_messages is None:
        history_messages = []

    if enable_cot:
        logger.debug("enable_cot=True is not supported for ollama and will be ignored.")

    #stream = bool(kwargs.pop("stream", False))
    stream = True
    user_stream = bool(kwargs.pop("stream", False))
    force_stream = False

    # Remove unsupported / irrelevant args
    kwargs.pop("max_tokens", None)
    kwargs.pop("hashing_kv", None)
    kwargs.pop("options", None)

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    if timeout in (0, None):
        timeout = 300

    api_key = kwargs.pop("api_key", None) or os.getenv("OLLAMA_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = _coerce_host_for_cloud_model(host, model)

    ollama_client = ollama.AsyncClient(
        host=host,
        timeout=timeout,
        headers=headers,
    )

    # 🔑 Decision rule: use generate() for instruction / RAG / JSON tasks
    use_generate = (
        kwargs.get("format") == "json"
        or system_prompt is not None
        or "extract" in prompt.lower()
        or "json" in prompt.lower()
    )

    try:
        if use_generate:
            force_stream = True
            # ----- GENERATE PATH (RAG / JSON / extraction) -----
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            #print("\n===== FINAL PROMPT SENT TO MODEL =====\n")
            #print(full_prompt[:5000])  # truncate if huge
            #print("\n===== END PROMPT =====\n")

            response = await ollama_client.generate(
                model=model,
                prompt=full_prompt,
                format=kwargs.get("format"),
                stream=stream,
            )

            if stream:
                async def inner():
                    async for chunk in response:
                        yield chunk["response"]
                # if not user_stream:
                return await collect_async_generator(inner())
#                return inner()
            else:
                return response["response"]

        else:
            # ----- CHAT PATH (pure conversation only) -----
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
            full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            print(full_prompt[:5000])  # truncate if huge
            print("\n===== END PROMPT =====\n")

            response = await ollama_client.chat(
                model=model,
                messages=messages,
                stream=stream,
            )

            if stream:
                async def inner():
                    async for chunk in response:
                        yield chunk["message"]["content"]
#                return inner()
                # if not user_stream:
                return await collect_async_generator(inner())
            else:
                return response["message"]["content"]

    finally:
        try:
            await ollama_client._client.aclose()
        except Exception:
            pass

async def ollama_model_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:

    if history_messages is None:
        history_messages = []

    if keyword_extraction:
        kwargs["format"] = "json"

    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    return await _ollama_model_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, ResponseError)
    ),
)
@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def ollama_embed(
    texts: list[str], embed_model: str = "bge-m3:latest", **kwargs
) -> np.ndarray:
    api_key = kwargs.pop("api_key", None)
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    
    # Dynamic timeout based on batch size: larger batches need more time
    # Base timeout is 30s, add 10s per 50 texts (conservative estimate)
    if timeout is None:
        batch_size_factor = max(1, len(texts) // 50)
        timeout = 30 + (batch_size_factor * 10)
        logger.debug(f"Dynamic timeout for {len(texts)} texts: {timeout}s")

    host = _coerce_host_for_cloud_model(host, embed_model)

    # Process in batches to avoid overwhelming the Ollama server
    # This is critical for large indexing operations that send hundreds of texts
    MAX_BATCH_SIZE = 100  # Process max 100 texts per request
    
    if len(texts) <= MAX_BATCH_SIZE:
        # Small batch: process directly
        ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
        try:
            options = kwargs.pop("options", {})
            logger.debug(f"Embedding batch of {len(texts)} texts (single request)")
            data = await ollama_client.embed(
                model=embed_model, input=texts, options=options
            )
            return np.array(data["embeddings"])
        except Exception as e:
            logger.error(f"Error in ollama_embed: {str(e)}")
            raise e
        finally:
            try:
                await ollama_client._client.aclose()
                logger.debug("Successfully closed Ollama client")
            except Exception as close_error:
                logger.debug(f"Note: Failed to close client: {close_error}")
    else:
        # Large batch: split into smaller chunks
        logger.info(f"Splitting {len(texts)} texts into batches of {MAX_BATCH_SIZE}")
        all_embeddings = []
        options = kwargs.pop("options", {})
        
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]
            batch_num = i // MAX_BATCH_SIZE + 1
            logger.debug(f"Processing batch {batch_num}: {len(batch)} texts")
            
            ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
            try:
                data = await ollama_client.embed(
                    model=embed_model, input=batch, options=options
                )
                batch_embeddings = data.get("embeddings", [])
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Batch {batch_num} complete: {len(batch_embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Error embedding batch {batch_num}: {str(e)}")
                raise e
            finally:
                try:
                    await ollama_client._client.aclose()
                except:
                    pass
        
        logger.info(f"Successfully embedded all {len(all_embeddings)} texts")
        return np.array(all_embeddings)
