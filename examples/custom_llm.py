"""Example usage of GraphRAG with custom LLM and Embedding services compatible with the OpenAI API."""

from typing import List

import instructor
from dotenv import load_dotenv

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService

load_dotenv()

DOMAIN = ""
QUERIES: List[str] = []
ENTITY_TYPES: List[str] = []

working_dir = "./examples/ignore/hp"
grag = GraphRAG(
    working_dir=working_dir,
    domain=DOMAIN,
    example_queries="\n".join(QUERIES),
    entity_types=ENTITY_TYPES,
    config=GraphRAG.Config(
        llm_service=OpenAILLMService(
            model="your-llm-model",
            base_url="llm.api.url.com",
            api_key="your-api-key",
            mode=instructor.Mode.JSON,
            api_version="your-llm-api_version",
            client="openai or azure"
        ),
        embedding_service=OpenAIEmbeddingService(
            model="your-embedding-model",
            base_url="emb.api.url.com",
            api_key="your-api-key",
            embedding_dim=512,  # the output embedding dim of the chosen model
            api_version="your-llm-api_version",
            client="openai or azure"
        ),
    ),
)
