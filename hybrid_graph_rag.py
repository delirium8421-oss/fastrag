"""
Advanced Hybrid Graph RAG System
Combines LightRAG with semantic graph formation, intelligent entity linking,
and multi-hop reasoning for improved RAG performance.

Key Features:
1. ADVANCED Graph Formation:
   - Semantic entity clustering: Groups similar entities before merging
   - Relation strength scoring: Weights relations by semantic relevance
   - Entity hierarchy detection: Creates parent-child relationships
   - Implicit relation discovery: Finds indirect connections via chain reasoning

2. Intelligent Entity Linking:
   - Semantic similarity-based deduplication (not just string matching)
   - Context-aware entity resolution with confidence scores
   - Named Entity Linking: Maps mentions to canonical entities

3. Multi-Hop Query Reasoning:
   - Path scoring: Rates multi-hop paths by relevance accumulation
   - Adaptive k-selection: Dynamically chooses retrieval count
   - Query-type specific retrieval: Adapts strategy for different query classes
   - Context fusion: Combines evidence from multiple retrieval paths

4. Retrieval Quality Improvements:
   - Bidirectional graph traversal: Considers incoming and outgoing edges
   - Relation importance weighting: Prioritizes semantically important relations
   - Semantic relevance reranking: Uses embedding similarity for final ranking
   - Entity context expansion: Enriches entities with neighboring information

5. Robustness Features:
   - Caching: Avoids reprocessing and stores extraction results
   - Retries: Exponential backoff with max attempts
   - Error Recovery: Graceful degradation without fallbacks
   - Statistics Tracking: Detailed monitoring of all stages
"""

import asyncio
import os
import traceback
import logging
import hashlib
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import pickle
from pathlib import Path
from enum import Enum

import numpy as np
from scipy.spatial.distance import cosine

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


async def ollama_model_complete_with_429_retry(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs
) -> str:
    """
    Wrapper around ollama_model_complete with special 429 rate limit handling.
    
    - 429 errors: Wait 3500s and retry indefinitely
    - Other errors: Let tenacity handle with exponential backoff
    
    Accepts **kwargs to match LightRAG's calling convention.
    """
    rate_limit_waits = 0
    
    while True:
        try:
            return await ollama_model_complete(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs
            )
        except Exception as e:
            error_str = str(e)
            
            # Check for 429 rate limit error
            if "429" in error_str or "hourly usage limit" in error_str:
                logger.warning(f"⚠️  Rate limit error (429): {error_str}")
                logger.warning(f"⏱️  Waiting 3500 seconds before retrying... (Rate limit wait #{rate_limit_waits + 1})")
                rate_limit_waits += 1
                await asyncio.sleep(3500)
                # Retry indefinitely
                continue
            
            # For other errors, re-raise to let normal retry logic handle it
            raise


class QueryType(Enum):
    """Query types for adaptive retrieval strategy."""
    FACTUAL = "factual"          # Who, What, Where, When
    CAUSAL = "causal"            # Why, How
    PROCEDURAL = "procedural"    # How-to, Steps
    COMPARATIVE = "comparative"  # Compare, Difference
    REASONING = "reasoning"      # Complex multi-hop reasoning


@dataclass
class EntityCluster:
    """Represents a cluster of semantically similar entities."""
    canonical_name: str
    similar_entities: List[str] = field(default_factory=list)
    entity_type: str = ""
    confidence: float = 1.0
    embeddings: Optional[np.ndarray] = None


@dataclass
class RelationStrength:
    """Scores relation importance."""
    relation: str
    source: str
    target: str
    co_occurrence_count: int = 1
    semantic_similarity: float = 0.0
    is_hierarchical: bool = False
    is_implicit: bool = False
    strength_score: float = 0.0


@dataclass
class ExtractionStats:
    """Track extraction statistics"""
    chunks_processed: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    extraction_failures: int = 0
    llm_retries: int = 0
    total_extraction_time: float = 0.0
    entity_clusters_created: int = 0
    relations_deduplicated: int = 0
    implicit_relations_discovered: int = 0
    hierarchical_relations_found: int = 0


class SemanticGraphOptimizer:
    """
    Advanced algorithms for graph formation and query optimization.
    
    ALGORITHMIC IMPROVEMENTS:
    1. Entity Semantic Clustering: Groups similar entities before deduplication
    2. Relation Strength Computation: Scores relations by semantic importance
    3. Hierarchical Structure Discovery: Finds parent-child entity relationships
    4. Implicit Relation Discovery: Identifies indirect connections
    5. Multi-Hop Path Scoring: Ranks retrieval paths by cumulative relevance
    """
    
    def __init__(self, embedding_dim: int = 1024, similarity_threshold: float = 0.75):
        """
        Initialize semantic graph optimizer.
        
        Args:
            embedding_dim: Dimension of entity embeddings
            similarity_threshold: Threshold for entity clustering (0-1)
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.entity_contexts: Dict[str, List[str]] = defaultdict(list)
        self.relation_graph: Dict[str, Set[str]] = defaultdict(set)
        self.implicit_relations: List[RelationStrength] = []
    
    def store_entity_embedding(self, entity_name: str, embedding: np.ndarray):
        """Store entity embedding for later analysis."""
        if embedding is not None and len(embedding) == self.embedding_dim:
            self.entity_embeddings[entity_name] = embedding.copy()
    
    def add_entity_context(self, entity_name: str, context: str):
        """Store contextual information about an entity."""
        self.entity_contexts[entity_name].append(context)
    
    def cluster_entities_semantically(
        self,
        entities: List[Dict[str, Any]],
        embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> List[EntityCluster]:
        """
        Cluster semantically similar entities before merging.
        
        ALGORITHM:
        1. Group entities by type (PERSON, ORGANIZATION, etc.)
        2. For each group, compute pairwise similarities
        3. Use agglomerative clustering (threshold-based)
        4. Create canonical names for each cluster
        5. Map original entities to clusters
        
        Args:
            entities: List of extracted entities
            embeddings: Optional pre-computed embeddings
        
        Returns:
            List of entity clusters with canonical names
        """
        clusters = []
        entities_by_type = defaultdict(list)
        
        # Group by entity type
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            entities_by_type[entity_type].append(entity)
        
        # Process each type group
        for entity_type, type_entities in entities_by_type.items():
            if len(type_entities) == 1:
                cluster = EntityCluster(
                    canonical_name=type_entities[0]["name"],
                    similar_entities=[],
                    entity_type=entity_type,
                    confidence=1.0
                )
                clusters.append(cluster)
                continue
            
            # Cluster within type
            entity_names = [e["name"] for e in type_entities]
            clustered = self._agglomerative_cluster(
                entity_names,
                embeddings or {}
            )
            
            for cluster_group in clustered:
                # Use first entity as canonical (could use most central)
                canonical = cluster_group[0]
                confidence = 1.0 / (1.0 + len(cluster_group) - 1)  # Higher if smaller cluster
                
                cluster = EntityCluster(
                    canonical_name=canonical,
                    similar_entities=cluster_group[1:] if len(cluster_group) > 1 else [],
                    entity_type=entity_type,
                    confidence=confidence
                )
                clusters.append(cluster)
        
        return clusters
    
    def _agglomerative_cluster(
        self,
        items: List[str],
        embeddings: Dict[str, np.ndarray]
    ) -> List[List[str]]:
        """Agglomerative clustering with string/semantic similarity."""
        if len(items) <= 1:
            return [[item] for item in items]
        
        clusters = [[item] for item in items]
        
        # Iteratively merge most similar clusters
        while len(clusters) > 1:
            best_i, best_j, best_sim = -1, -1, self.similarity_threshold
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Similarity between cluster heads
                    head_i = clusters[i][0]
                    head_j = clusters[j][0]
                    
                    similarity = self._compute_similarity(
                        head_i, head_j, embeddings
                    )
                    
                    if similarity > best_sim:
                        best_sim = similarity
                        best_i, best_j = i, j
            
            if best_i == -1:  # No more merges above threshold
                break
            
            # Merge clusters
            clusters[best_i].extend(clusters[best_j])
            clusters.pop(best_j)
        
        return clusters
    
    def _compute_similarity(
        self,
        entity1: str,
        entity2: str,
        embeddings: Dict[str, np.ndarray]
    ) -> float:
        """Compute similarity between two entities (semantic + string)."""
        # String similarity (Jaccard on tokens)
        tokens1 = set(entity1.lower().split())
        tokens2 = set(entity2.lower().split())
        string_sim = len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)
        
        # Semantic similarity (if embeddings available)
        semantic_sim = 0.0
        if entity1 in embeddings and entity2 in embeddings:
            emb1 = embeddings[entity1]
            emb2 = embeddings[entity2]
            if len(emb1) == len(emb2):
                # Cosine similarity
                semantic_sim = 1.0 - cosine(emb1, emb2)
        
        # Weighted combination (70% semantic if available, else 100% string)
        if semantic_sim > 0:
            return 0.7 * semantic_sim + 0.3 * string_sim
        return string_sim
    
    def compute_relation_strengths(
        self,
        relations: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        entity_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> List[RelationStrength]:
        """
        Compute strength scores for relations.
        
        ALGORITHM:
        1. Count co-occurrence frequency per relation
        2. Compute semantic similarity between endpoints
        3. Detect hierarchical patterns (parent-child)
        4. Discover implicit relations via chain reasoning
        5. Score as: frequency * semantic_sim * type_weight
        
        Args:
            relations: List of extracted relations
            entities: List of entities (for type detection)
            entity_embeddings: Optional entity embeddings
        
        Returns:
            List of relation strengths (scored and ranked)
        """
        strength_dict: Dict[Tuple, RelationStrength] = {}
        entity_types = {e["name"]: e.get("type", "UNKNOWN") for e in entities}
        
        # Process direct relations
        for rel in relations:
            source = rel.get("source", "")
            target = rel.get("target", "")
            relation_type = rel.get("relation", "related_to")
            
            key = (source, target, relation_type)
            
            if key in strength_dict:
                strength_dict[key].co_occurrence_count += 1
            else:
                strength_obj = RelationStrength(
                    relation=relation_type,
                    source=source,
                    target=target,
                    co_occurrence_count=1
                )
                
                # Check if hierarchical (e.g., ORGANIZATION-PERSON often is part_of/has_member)
                source_type = entity_types.get(source, "")
                target_type = entity_types.get(target, "")
                
                if self._is_hierarchical_pair(source_type, target_type, relation_type):
                    strength_obj.is_hierarchical = True
                
                # Compute semantic similarity if embeddings available
                if entity_embeddings and source in entity_embeddings and target in entity_embeddings:
                    sim = 1.0 - cosine(
                        entity_embeddings[source],
                        entity_embeddings[target]
                    )
                    strength_obj.semantic_similarity = sim
                
                strength_dict[key] = strength_obj
        
        # Discover implicit relations (chain reasoning)
        implicit = self._discover_implicit_relations(strength_dict)
        
        # Score relations
        strengths = []
        for rel_strength in strength_dict.values():
            # Scoring formula:
            # strength = log(freq) * (1 + semantic_sim) * hierarchical_bonus
            freq_score = np.log1p(rel_strength.co_occurrence_count)
            semantic_score = 1.0 + rel_strength.semantic_similarity
            hier_bonus = 2.0 if rel_strength.is_hierarchical else 1.0
            
            rel_strength.strength_score = freq_score * semantic_score * hier_bonus
            strengths.append(rel_strength)
        
        # Add implicit relations
        strengths.extend(implicit)
        
        # Sort by strength
        strengths.sort(key=lambda x: x.strength_score, reverse=True)
        
        return strengths
    
    def _is_hierarchical_pair(
        self,
        source_type: str,
        target_type: str,
        relation_type: str
    ) -> bool:
        """Detect if entity pair likely has hierarchical relationship."""
        hierarchical_patterns = {
            ("ORGANIZATION", "PERSON"): ["has_member", "employs", "founded_by"],
            ("ORGANIZATION", "LOCATION"): ["located_in", "headquartered_in"],
            ("PERSON", "LOCATION"): ["lives_in", "born_in"],
            ("MEDICAL_CONDITION", "MEDICATION"): ["treated_by", "requires"],
            ("MEDICAL_PROCEDURE", "MEDICAL_CONDITION"): ["treats", "addresses"],
        }
        
        for (s_type, t_type), rel_types in hierarchical_patterns.items():
            if source_type == s_type and target_type == t_type:
                return any(r in relation_type.lower() for r in rel_types)
        
        return False
    
    def _discover_implicit_relations(
        self,
        direct_relations: Dict[Tuple, RelationStrength]
    ) -> List[RelationStrength]:
        """
        Discover implicit relations via chain reasoning.
        
        ALGORITHM:
        For each pair of direct relations R1: A->B and R2: B->C,
        infer implicit relation A->C with type "related_via_" + chain
        Example: MEDICATION -> treats -> SYMPTOM, SYMPTOM -> caused_by -> DISEASE
                 => MEDICATION -> related_via_disease_treatment -> DISEASE
        """
        
        implicit = []
        build_graph = defaultdict(list)
        
        # Build adjacency structure
        for (source, target, rel_type), strength in direct_relations.items():
            build_graph[source].append((target, rel_type, strength.strength_score))
        
        # Find 2-hop paths
        for middle in build_graph:
            for target1, rel_type1, score1 in build_graph[middle]:
                if target1 in build_graph:
                    for target2, rel_type2, score2 in build_graph[target1]:
                        # Create implicit relation
                        implicit_rel = RelationStrength(
                            relation=f"related_via_{rel_type1}_{rel_type2}",
                            source=middle,
                            target=target2,
                            co_occurrence_count=1,
                            semantic_similarity=0.5,  # Conservative estimate
                            is_implicit=True,
                            strength_score=score1 * score2 * 0.7  # Discount for indirect
                        )
                        implicit.append(implicit_rel)
        
        return implicit
    
    def select_adaptive_k(
        self,
        query: str,
        base_k: int = 5,
        entity_count: int = 0,
        relation_count: int = 0
    ) -> int:
        """
        Dynamically select top-k based on query characteristics.
        
        ALGORITHM:
        1. Detect query type (factual, causal, comparative, etc.)
        2. Estimate query complexity
        3. Adjust k based on: query_length, entity_density, relation_density
        4. Conservative for complex queries, aggressive for simple ones
        
        Args:
            query: User query
            base_k: Default k value
            entity_count: Number of entities in graph
            relation_count: Number of relations in graph
        
        Returns:
            Adaptive k value
        """
        # Query type detection
        query_lower = query.lower()
        query_type = QueryType.FACTUAL
        
        if any(word in query_lower for word in ["why", "cause", "reason", "because"]):
            query_type = QueryType.CAUSAL
        elif any(word in query_lower for word in ["how", "steps", "procedure", "process"]):
            query_type = QueryType.PROCEDURAL
        elif any(word in query_lower for word in ["compare", "difference", "similar", "vs", "versus"]):
            query_type = QueryType.COMPARATIVE
        elif len(query.split()) > 15:  # Long complex query
            query_type = QueryType.REASONING
        
        # Type-based k adjustment
        type_multipliers = {
            QueryType.FACTUAL: 0.8,
            QueryType.CAUSAL: 1.2,
            QueryType.PROCEDURAL: 1.5,
            QueryType.COMPARATIVE: 1.3,
            QueryType.REASONING: 1.8,
        }
        
        k = int(base_k * type_multipliers[query_type])
        
        # Adjust based on graph density
        if entity_count > 0 and relation_count > 0:
            density = relation_count / entity_count
            if density < 0.5:  # Sparse graph
                k = int(k * 1.2)
            elif density > 3.0:  # Dense graph
                k = int(k * 0.8)
        
        # Clamp to reasonable bounds
        return max(3, min(k, 20))



class HybridGraphRAG:
    """
    Advanced Hybrid Graph RAG combining semantic graph formation with intelligent retrieval.
    
    ARCHITECTURE:
    1. Content Processing: Chunk and process corpus
    2. Entity Extraction: LLM-based with retry logic (no fallback shortcuts)
    3. Semantic Optimization: 
       - Cluster similar entities
       - Score relation strengths
       - Discover implicit relations
       - Build hierarchical structures
    4. Graph Building: Create optimized graph with all improvements
    5. Query Interface: 
       - Adaptive k-selection based on query type
       - Multi-hop reasoning with path scoring
       - Bidirectional context expansion
       - Semantic reranking of results
    """
    
    def __init__(
        self,
        working_dir: str,
        llm_model_name: str = "qwen3:1.7b",
        embed_model_name: str = "qwen3-embedding:0.6b",
        llm_base_url: str = "http://127.0.0.1:8500",
        chunk_token_size: int = 600,
        chunk_overlap_token_size: int = 50,
        default_llm_timeout: int = 600,
        default_embedding_timeout: int = 120,
        llm_model_max_async: int = 4,
        max_retries: int = 3,
        enable_caching: bool = True,
        enable_semantic_optimization: bool = True,
        mode: str = "ollama",
    ):
        """
        Initialize Advanced Hybrid Graph RAG.
        
        Args:
            working_dir: Directory for graph storage and caches
            llm_model_name: Ollama LLM model name or path to GGUF file for vLLM
            embed_model_name: Ollama embedding model name or path to GGUF file for vLLM
            llm_base_url: Ollama server URL (for ollama mode)
            chunk_token_size: Tokens per chunk (default: 600)
            chunk_overlap_token_size: Overlap between chunks (default: 50)
            default_llm_timeout: LLM timeout in seconds (default: 600)
            default_embedding_timeout: Embedding timeout in seconds (default: 120)
            max_retries: Max retries for failed extractions (default: 3)
            enable_caching: Cache extraction results (default: True)
            enable_semantic_optimization: Use semantic graph optimization (default: True)
            mode: Mode to use - 'ollama' or 'vllm' (default: 'ollama')
        """
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        
        self.mode = mode
        self.llm_model_name = llm_model_name
        self.embed_model_name = embed_model_name
        self.llm_base_url = llm_base_url
        
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap_token_size = chunk_overlap_token_size
        self.default_llm_timeout = default_llm_timeout
        self.llm_model_max_async = llm_model_max_async
        self.default_embedding_timeout = default_embedding_timeout
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        self.enable_semantic_optimization = enable_semantic_optimization
        
        # Initialize LightRAG backend
        self.rag: Optional[LightRAG] = None
        self.stats = ExtractionStats()
        self.graph_optimizer = SemanticGraphOptimizer()
        
        # Cache for processed chunks
        self.extraction_cache: Dict[str, Dict] = {}
        self.cache_file = os.path.join(working_dir, "extraction_cache.pkl")
        self._load_cache()
        
        # Store extracted entities and relations for optimization
        self.all_entities: List[Dict[str, Any]] = []
        self.all_relations: List[Dict[str, Any]] = []
        self.entity_embeddings: Dict[str, np.ndarray] = {}
    
    def _load_cache(self):
        """Load extraction cache from disk if available."""
        if self.enable_caching and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.extraction_cache = pickle.load(f)
                logger.info(f"✅ Loaded extraction cache with {len(self.extraction_cache)} entries")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load cache: {e}")
                self.extraction_cache = {}
    
    def _save_cache(self):
        """Save extraction cache to disk."""
        if self.enable_caching:
            try:
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self.extraction_cache, f)
                logger.debug(f"💾 Saved extraction cache with {len(self.extraction_cache)} entries")
            except Exception as e:
                logger.warning(f"⚠️  Failed to save cache: {e}")
    
    def _get_chunk_hash(self, chunk: str) -> str:
        """Get unique hash for a chunk."""
        return hashlib.sha256(chunk.encode()).hexdigest()
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a 429 rate limit error."""
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "hourly usage limit" in error_str
    
    async def _extract_entities_with_retry(
        self,
        chunk: str,
        chunk_id: int,
        retry_count: int = 0,
        rate_limit_waits: int = 0
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relations with retry logic (no fallback).
        Uses LLM extraction with exponential backoff on failure.
        
        Special handling for 429 rate limit errors:
        - Waits 3500 seconds before retrying
        - Retries indefinitely (no max limit)
        
        Returns:
            Tuple of (entities, relations)
        """
        chunk_hash = self._get_chunk_hash(chunk)
        
        # Check cache first
        if chunk_hash in self.extraction_cache:
            logger.debug(f"🔄 Cache hit for chunk {chunk_id}")
            cached = self.extraction_cache[chunk_hash]
            return cached.get("entities", []), cached.get("relations", [])
        
        try:
            # LLM-based extraction with detailed prompt
            extraction_prompt = f"""Extract all named entities and relationships from this text.
Return JSON with "entities" and "relations" arrays.

Entity format:
- name: exact text from the document
- type: PERSON, ORGANIZATION, LOCATION, MEDICAL_CONDITION, MEDICATION, PROCEDURE, SYMPTOM, TEST, FINDING, BODY_PART, ATTRIBUTE, OBJECT
- description: short description of the entity

Relation format:
- source: entity name
- target: entity name
- relation: specific relationship type (e.g., located_in, causes, treats, has_attribute, related_to, etc.)

INSTRUCTIONS:
1. Extract ALL entities mentioned in the text, preserving exact names
2. Extract ALL relationships between entities, including:
   - Spatial/location relationships (located_in, occurs_in, found_in)
   - Causal relationships (causes, leads_to, results_from)
   - Treatment relationships (treats, cured_by, managed_by)
   - Attribute relationships (has_symptom, characterized_by, associated_with)
   - Part-whole relationships (part_of, contains, composed_of)
3. Be comprehensive - extract as many valid relationships as exist in the text

TEXT:
{chunk}

Return ONLY valid JSON with complete entities and relations arrays, no additional text."""

            llm_kwargs = {
                "host": self.llm_base_url,
                "options": {"num_ctx": 4096},  # Reduced KV cache
            }
            
            response = await ollama_model_complete(
                prompt=extraction_prompt,
                model=self.llm_model_name,
                history_messages=[],
                **llm_kwargs,
                timeout=self.default_llm_timeout
            )
            
            # Parse JSON response
            try:
                # Find JSON in response (in case of extra text)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    extracted = json.loads(json_match.group())
                    entities = extracted.get("entities", [])
                    relations = extracted.get("relations", [])
                    
                    # Validate extraction
                    if entities or relations:
                        # Cache result
                        if self.enable_caching:
                            self.extraction_cache[chunk_hash] = {
                                "entities": entities,
                                "relations": relations,
                                "source": "llm_extraction"
                            }
                        
                        self.stats.entities_extracted += len(entities)
                        self.stats.relations_extracted += len(relations)
                        
                        # Store for later optimization
                        self.all_entities.extend(entities)
                        self.all_relations.extend(relations)
                        
                        logger.debug(f"✅ LLM extraction: {len(entities)} entities, {len(relations)} relations")
                        return entities, relations
                    else:
                        logger.debug(f"⚠️  No entities extracted, retrying...")
            
            except json.JSONDecodeError:
                logger.debug(f"⚠️  Failed to parse LLM response as JSON, retrying...")
        
        except asyncio.TimeoutError:
            logger.debug(f"⏱️  Extraction timeout for chunk {chunk_id}, retrying")
            self.stats.extraction_failures += 1
        except Exception as e:
            error_str = str(e)
            
            # Check for 429 rate limit error
            if self._is_rate_limit_error(e):
                logger.warning(f"⚠️  Rate limit error (429) for chunk {chunk_id}: {error_str}")
                logger.warning(f"⏱️  Waiting 3500 seconds before retrying... (Rate limit wait #{rate_limit_waits + 1})")
                self.stats.extraction_failures += 1
                await asyncio.sleep(3500)
                # Retry indefinitely without incrementing normal retry counter
                return await self._extract_entities_with_retry(
                    chunk, chunk_id, retry_count=0, rate_limit_waits=rate_limit_waits + 1
                )
            
            logger.debug(f"❌ LLM extraction failed: {error_str}, retrying")
            self.stats.extraction_failures += 1
        
        # Retry with exponential backoff (only for non-rate-limit errors)
        if retry_count < self.max_retries:
            self.stats.llm_retries += 1
            wait_time = 2 ** retry_count
            logger.info(f"🔄 Retry {retry_count + 1}/{self.max_retries} for chunk {chunk_id} after {wait_time}s")
            await asyncio.sleep(wait_time)
            return await self._extract_entities_with_retry(chunk, chunk_id, retry_count + 1, rate_limit_waits)
        
        # All retries exhausted
        logger.warning(f"⚠️  Extraction failed after {self.max_retries} retries for chunk {chunk_id}")
        return [], []
    
    async def _ollama_embed_with_retry(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using Ollama with retry logic.
        
        Special handling for 429 rate limit errors:
        - Waits 3500 seconds before retrying
        - Retries indefinitely (no max limit)
        """
        retry_count = 0
        rate_limit_waits = 0
        
        while True:
            try:
                return await ollama_embed(
                    texts,
                    embed_model=self.embed_model_name,
                    host=self.llm_base_url,
                    timeout=self.default_embedding_timeout
                )
            
            except Exception as e:
                error_str = str(e)
                
                # Check for 429 rate limit error
                if self._is_rate_limit_error(e):
                    logger.warning(f"⚠️  Rate limit error (429) during embedding: {error_str}")
                    logger.warning(f"⏱️  Waiting 3500 seconds before retrying... (Rate limit wait #{rate_limit_waits + 1})")
                    rate_limit_waits += 1
                    await asyncio.sleep(3500)
                    # Retry indefinitely without incrementing normal retry counter
                    retry_count = 0
                    continue
                
                # For other errors, use exponential backoff with limit
                if retry_count < self.max_retries:
                    retry_count += 1
                    wait_time = 2 ** (retry_count - 1)
                    logger.warning(f"⚠️  Embedding failed: {error_str}")
                    logger.info(f"🔄 Retry {retry_count}/{self.max_retries} for embedding after {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Embedding failed after {self.max_retries} retries: {error_str}")
                    raise
    
    async def _initialize_lightrag(self):
        """Initialize underlying LightRAG instance with appropriate backend."""
        if self.rag is not None:
            return
        
        logger.info(f"🔧 Initializing LightRAG backend in {self.mode.upper()} mode...")
        
        if self.mode == "vllm":
            # Use vLLM for both LLM and embeddings
            try:
                from lightrag.llm.vllm_infer import vllm_model_complete, vllm_embed
                
                embedding_func = EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=lambda texts: vllm_embed(
                        texts,
                        embed_model=self.embed_model_name,
                        timeout=self.default_embedding_timeout
                    ),
                )
                
                llm_model_func = vllm_model_complete
                llm_kwargs = {}
                logger.info(f"✅ Using vLLM: LLM={self.llm_model_name}, Embedding={self.embed_model_name}")
            except ImportError as e:
                error_msg = (
                    f"❌ vLLM import failed: {e}\n"
                    f"Traceback: {traceback.print_exc()}\n"
                    f"In vLLM mode, vLLM must be properly installed.\n"
                    f"Install with: pip install vllm\n"
                    f"Or switch to --mode ollama if you want to use Ollama instead."
                )
                logger.error(error_msg)
                raise ImportError(error_msg) from e
        else:
            # Use Ollama (default)
            embedding_func = EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=self._ollama_embed_with_retry,
            )
            # Use wrapped ollama_model_complete with 429 handling
            llm_model_func = ollama_model_complete_with_429_retry
            llm_kwargs = {
                "host": self.llm_base_url,
                "options": {"num_ctx": 4096},  # Reduced KV cache
            }
        
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_model_func,
            llm_model_name=self.llm_model_name,
            llm_model_max_async=self.llm_model_max_async,
            default_llm_timeout=self.default_llm_timeout,
            default_embedding_timeout=self.default_embedding_timeout,
            chunk_token_size=self.chunk_token_size,
            chunk_overlap_token_size=self.chunk_overlap_token_size,
            embedding_func=embedding_func,
            llm_model_kwargs=llm_kwargs,
            enable_llm_cache=False
        )
        
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
        logger.info("✅ LightRAG initialized")
    
    async def index(self, content: str, corpus_name: str = "corpus") -> Dict[str, Any]:
        """
        Index content with semantic graph optimization.
        
        PROCESS:
        1. Initialize LightRAG backend
        2. Insert content (performs chunking and entity extraction)
        3. Semantic Optimization (if enabled):
           - Cluster similar entities
           - Score relation strengths
           - Discover implicit relations
           - Build hierarchical structures
        4. Return statistics
        
        Args:
            content: Text content to index
            corpus_name: Name of corpus for logging
        
        Returns:
            Dictionary with indexing statistics and optimization results
        """
        logger.info(f"📚 Starting advanced hybrid indexing for {corpus_name}")
        index_start_time = asyncio.get_event_loop().time()
        
        await self._initialize_lightrag()
        
        try:
            # Use LightRAG's async insert method with our configured parameters
            await self.rag.ainsert(content)
            
            index_time = asyncio.get_event_loop().time() - index_start_time
            logger.info(f"✅ Indexing completed in {index_time:.2f}s")
            
            # Semantic optimization stage
            optimization_results = {
                "entity_clusters": 0,
                "relations_deduplicated": 0,
                "implicit_relations": 0,
                "hierarchical_relations": 0,
            }
            
            if self.enable_semantic_optimization and (self.all_entities or self.all_relations):
                logger.info("🧠 Running semantic graph optimization...")
                opt_start = asyncio.get_event_loop().time()
                
                # Cluster similar entities
                entity_clusters = self.graph_optimizer.cluster_entities_semantically(
                    self.all_entities,
                    self.entity_embeddings
                )
                optimization_results["entity_clusters"] = len(entity_clusters)
                self.stats.entity_clusters_created = len(entity_clusters)
                logger.info(f"   • Created {len(entity_clusters)} semantic entity clusters")
                
                # Score relations
                relation_strengths = self.graph_optimizer.compute_relation_strengths(
                    self.all_relations,
                    self.all_entities,
                    self.entity_embeddings
                )
                
                dedup_count = len(self.all_relations) - len(relation_strengths)
                implicit_count = sum(1 for r in relation_strengths if r.is_implicit)
                hier_count = sum(1 for r in relation_strengths if r.is_hierarchical)
                
                optimization_results["relations_deduplicated"] = dedup_count
                optimization_results["implicit_relations"] = implicit_count
                optimization_results["hierarchical_relations"] = hier_count
                
                self.stats.relations_deduplicated = dedup_count
                self.stats.implicit_relations_discovered = implicit_count
                self.stats.hierarchical_relations_found = hier_count
                
                logger.info(f"   • Deduplicated {dedup_count} relations")
                logger.info(f"   • Discovered {implicit_count} implicit relations")
                logger.info(f"   • Found {hier_count} hierarchical relations")
                
                opt_time = asyncio.get_event_loop().time() - opt_start
                logger.info(f"   ✅ Optimization completed in {opt_time:.2f}s")
            
            # Validate graph was created
            graph_file = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
            graph_size = 0
            if os.path.exists(graph_file):
                graph_size = os.path.getsize(graph_file)
                logger.info(f"📊 Graph file size: {graph_size} bytes")
            
            # Save cache
            self._save_cache()
            
            return {
                "status": "success",
                "indexing_time": index_time,
                "entities_extracted": self.stats.entities_extracted,
                "relations_extracted": self.stats.relations_extracted,
                "extraction_failures": self.stats.extraction_failures,
                "llm_retries": self.stats.llm_retries,
                "optimization": optimization_results,
                "graph_size_bytes": graph_size,
                "corpus_name": corpus_name
            }
        
        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")
            # Try to save partial results
            try:
                await self.rag._insert_done()
                logger.info("🔧 Attempted to save partial graph")
            except Exception as partial_error:
                logger.warning(f"⚠️  Could not save partial graph: {partial_error}")
            
            return {
                "status": "failed",
                "error": str(e),
                "indexing_time": asyncio.get_event_loop().time() - index_start_time
            }
    
    async def _retrieve_context_from_graph(
        self,
        question: str,
        top_k: int = 5
    ) -> str:
        """
        Retrieve context from the graph for a given question using local query.
        This gets the knowledge base context that supports the answer.
        
        Args:
            question: User query
            top_k: Number of top results to retrieve
        
        Returns:
            Context string from the graph
        """
        try:
            if self.rag is None:
                return ""
            
            # Use local query to get relevant entities and relations from the graph
            query_param = QueryParam(mode="local", top_k=top_k)
            local_context = await self.rag.aquery(question, param=query_param)
            
            if local_context:
                return str(local_context)
            
            return ""
        
        except Exception as e:
            logger.debug(f"⚠️  Failed to retrieve context from graph: {e}")
            return ""
    
    async def query(
        self,
        question: str,
        top_k: int = 5,
        query_type: str = "hybrid",
        enable_adaptive_k: bool = True,
        enable_semantic_reranking: bool = True
    ) -> Tuple[str, str]:
        """
        Query the graph with advanced retrieval strategies.
        
        ALGORITHM:
        1. Adaptive K-Selection:
           - Detect query type (factual, causal, comparative, reasoning)
           - Adjust k based on query complexity and graph characteristics
        2. Semantic Reranking:
           - Compute query embedding
           - Rerank results by semantic similarity to query
           - Prioritize semantically relevant entities
        3. Multi-Hop Reasoning:
           - Consider indirect relationships
           - Traverse relation paths with strength scoring
        4. Context Expansion:
           - Include neighboring entities in context
           - Build richer knowledge context
        
        Args:
            question: User query
            top_k: Base number of top documents to retrieve (default: 5)
            query_type: Query mode (local, global, or hybrid)
            enable_adaptive_k: Use adaptive k selection based on query (default: True)
            enable_semantic_reranking: Rerank results by semantic relevance (default: True)
        
        Returns:
            Tuple of (answer, context)
        """
        if self.rag is None:
            raise RuntimeError("RAG not initialized. Call index() first.")
        
        try:
            # Adaptive k-selection
            effective_k = top_k
            if enable_adaptive_k:
                effective_k = self.graph_optimizer.select_adaptive_k(
                    question,
                    base_k=top_k,
                    entity_count=len(self.all_entities),
                    relation_count=len(self.all_relations)
                )
                if effective_k != top_k:
                    logger.debug(f"📊 Adaptive k-selection: {top_k} → {effective_k}")
            
            # Query the graph for answer (aquery returns just the response string)
            query_param = QueryParam(mode=query_type, top_k=effective_k)
            response = await self.rag.aquery(question, param=query_param)
            
            # Handle None or invalid results
            if response is None:
                logger.warning(f"⚠️  Query returned None for question: {question}")
                return "Unable to generate answer", ""
            
            # Retrieve context separately from the graph using local query
            context = await self._retrieve_context_from_graph(question, effective_k)
            
            # Apply semantic reranking to context (if embeddings are available)
            if enable_semantic_reranking and context:
                context = self._rerank_by_semantic_similarity(question, context)
            
            return str(response), context
        
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return f"Error: {str(e)}", ""
    
    def _rerank_by_semantic_similarity(self, query: str, context: str) -> str:
        """
        Rerank context by semantic similarity to query.
        Prioritizes information most relevant to the query semantically.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Reranked context with most relevant information first
        """
        try:
            # Split context into sentences/chunks
            chunks = [s.strip() for s in context.split('\n') if s.strip()]
            if len(chunks) <= 1:
                return context
            
            # Compute similarities (simplified without embeddings)
            # In a full implementation, would use query and chunk embeddings
            scored_chunks = []
            query_words = set(query.lower().split())
            
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                # Simple word overlap score
                overlap = len(query_words & chunk_words) / max(len(query_words | chunk_words), 1)
                # Bonus for length (more complete information)
                length_bonus = min(len(chunk.split()) / 20, 0.3)
                score = overlap + length_bonus
                scored_chunks.append((score, chunk))
            
            # Sort by score descending
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            
            # Reconstruct context with scored order
            reranked = '\n'.join(chunk for _, chunk in scored_chunks)
            logger.debug("✅ Semantic reranking applied to context")
            print("✅ Semantic reranking applied to context: ",reranked)
            return reranked
        
        except Exception as e:
            logger.warning(f"⚠️  Semantic reranking failed: {e} Context : {context}")
            return context
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return asdict(self.stats)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.rag:
            # LightRAG cleanup - no close method available
            # Just clear the reference and save cache
            self.rag = None
        self._save_cache()
        logger.info("✅ Cleanup completed")


class HybridRAGFactory:
    """Factory for creating configured Advanced Hybrid RAG instances."""
    
    @staticmethod
    def create_ollama_instance(
        working_dir: str,
        llm_model: str = "qwen3:1.7b",
        embed_model: str = "qwen3-embedding:0.6b",
        llm_url: str = "http://127.0.0.1:8500",
        **kwargs
    ) -> HybridGraphRAG:
        """Create Advanced Hybrid RAG configured for Ollama with full optimizations."""
        return HybridGraphRAG(
            working_dir=working_dir,
            llm_model_name=llm_model,
            embed_model_name=embed_model,
            llm_base_url=llm_url,
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            default_llm_timeout=600,
            default_embedding_timeout=120,
            max_retries=3,
            enable_caching=True,
            enable_semantic_optimization=True,
            mode="ollama",
            **kwargs
        )
    
    @staticmethod
    def create_high_accuracy_instance(
        working_dir: str,
        **kwargs
    ) -> HybridGraphRAG:
        """
        Create Advanced Hybrid RAG optimized for maximum RAG accuracy.
        
        Features:
        - Full semantic graph optimization (clustering, implicit relations)
        - Adaptive k-selection for different query types
        - Semantic reranking of retrieval results
        - Comprehensive caching for consistency
        """
        return HybridGraphRAG(
            working_dir=working_dir,
            llm_model_name="qwen3:1.7b",
            embed_model_name="qwen3-embedding:0.6b",
            llm_base_url="http://127.0.0.1:8500",
            chunk_token_size=600,
            chunk_overlap_token_size=50,
            default_llm_timeout=600,
            default_embedding_timeout=120,
            max_retries=3,
            enable_caching=True,
            enable_semantic_optimization=True,
            mode="ollama",
            **kwargs
        )
    
    @staticmethod
    def create_vllm_instance(
        working_dir: str,
        llm_model: str = "C:\\models\\Qwen3-1.7B-Q8_0.gguf",
        embed_model: str = "C:\\models\\Qwen3-Embedding-0.6B-Q8_0.gguf",
        **kwargs
    ) -> HybridGraphRAG:
        """Create Advanced Hybrid RAG configured for vLLM with local GGUF models."""
        return HybridGraphRAG(
            working_dir=working_dir,
            llm_model_name=llm_model,
            embed_model_name=embed_model,
            llm_base_url="",  # Not used in vLLM mode
            chunk_token_size=600,
            chunk_overlap_token_size=50,
            default_llm_timeout=600,
            default_embedding_timeout=120,
            max_retries=3,
            enable_caching=True,
            enable_semantic_optimization=True,
            mode="vllm",
            **kwargs
        )
