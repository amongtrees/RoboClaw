"""Semantic memory — long-term vector store backed by Milvus.

Collections:
- episode_embeddings: Semantic search over past task episodes
- skill_embeddings: Match skills to task descriptions
- failure_patterns: Retrieve known failure patterns during recovery
"""

from __future__ import annotations

from time import time
from typing import Any


class SemanticMemory:
    """Milvus-backed semantic memory for vector similarity search.

    Falls back to in-memory cosine similarity when Milvus is unavailable.
    """

    def __init__(self, milvus_config: Any = None) -> None:
        self._config = milvus_config
        self._connected = False
        # In-memory fallback
        self._collections: dict[str, list[dict[str, Any]]] = {
            "episode_embeddings": [],
            "skill_embeddings": [],
            "failure_patterns": [],
        }

    async def initialize(self) -> None:
        """Connect to Milvus and create collections if needed."""
        if self._config:
            try:
                from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

                connections.connect(
                    alias="default",
                    host=self._config.host,
                    port=self._config.port,
                    db_name=self._config.db_name,
                )
                self._connected = True
            except Exception:
                self._connected = False

    async def insert(
        self,
        collection_name: str,
        entry_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> str:
        """Insert a vector embedding with metadata into a collection."""
        entry = {
            "entry_id": entry_id,
            "embedding": embedding,
            "metadata": metadata,
            "timestamp": time(),
        }

        if self._connected:
            try:
                from pymilvus import Collection

                col = Collection(collection_name)
                col.insert([
                    [entry_id],
                    [embedding],
                    [metadata.get("summary", "")],
                    [int(time())],
                ])
                col.flush()
                return entry_id
            except Exception:
                pass

        # In-memory fallback
        if collection_name in self._collections:
            self._collections[collection_name].append(entry)
        return entry_id

    async def search(
        self,
        collection_name: str,
        query_embedding: list[float] | None = None,
        query_text: str = "",
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar entries in a collection.

        Uses cosine similarity. Falls back to in-memory computation.
        """
        if query_embedding is None:
            query_embedding = []

        if self._connected and query_embedding:
            try:
                from pymilvus import Collection

                col = Collection(collection_name)
                col.load()
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                results = col.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=self._build_filter_expr(filters),
                )
                return [
                    {
                        "entry_id": hit.id,
                        "score": float(hit.distance),
                        "metadata": hit.entity.get("summary", ""),
                    }
                    for hit in results[0]
                ]
            except Exception:
                pass

        # In-memory cosine similarity
        entries = self._collections.get(collection_name, [])
        if not query_embedding:
            return entries[:top_k]

        scored = []
        for entry in entries:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            scored.append({**entry, "score": score})

        if filters:
            scored = [
                s
                for s in scored
                if all(s.get("metadata", {}).get(k) == v for k, v in filters.items())
            ]

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # --- Convenience methods ---

    async def index_episode(self, episode_id: str, embedding: list[float], summary: str, task_type: str, outcome: str) -> str:
        """Index an episode for future semantic retrieval."""
        return await self.insert(
            "episode_embeddings",
            episode_id,
            embedding,
            {"summary": summary, "task_type": task_type, "outcome": outcome},
        )

    async def find_similar_episodes(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Find episodes similar to the given embedding."""
        return await self.search("episode_embeddings", query_embedding=embedding, top_k=top_k)

    async def find_failure_patterns(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Find known failure patterns similar to the current error context."""
        return await self.search("failure_patterns", query_embedding=embedding, top_k=top_k)

    async def find_skills(self, embedding: list[float], top_k: int = 10) -> list[dict[str, Any]]:
        """Find skills matching a task description embedding."""
        return await self.search("skill_embeddings", query_embedding=embedding, top_k=top_k)

    # --- Helpers ---

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _build_filter_expr(filters: dict[str, Any] | None) -> str | None:
        if not filters:
            return None
        parts = [f'{k} == "{v}"' for k, v in filters.items()]
        return " and ".join(parts)

    async def close(self) -> None:
        """Clean up Milvus connection."""
        if self._connected:
            try:
                from pymilvus import connections
                connections.disconnect("default")
            except Exception:
                pass
