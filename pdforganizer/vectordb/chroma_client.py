"""ChromaDB implementation of VectorDBClient."""

import logging
from typing import Any, Dict, List, Optional, cast

import chromadb
from chromadb.api import ClientAPI

from pdforganizer.config import AppConfig
from pdforganizer.vectordb import SearchResult, VectorDBClient

LOGGER = logging.getLogger(__name__)


class ChromaDBClient(VectorDBClient):
    """Local ChromaDB implementation."""

    def __init__(self, settings: AppConfig):
        self._settings = settings
        # Defaults to local persistent client as per current requirements
        LOGGER.info(f"🔌 Initializing Local ChromaDB at {settings.db_path}")
        self._client: ClientAPI = chromadb.PersistentClient(path=str(settings.db_path))

    def get_or_create_collection(self, name: str) -> Any:
        return self._client.get_or_create_collection(name=name)

    def upsert(
        self,
        collection_name: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        collection = self._client.get_collection(name=collection_name)
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
        )

    def update(
        self,
        collection_name: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        collection = self._client.get_collection(name=collection_name)
        collection.update(
            ids=ids,
            documents=documents,
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
        )

    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        collection = self._client.get_collection(name=collection_name)

        # Explicitly pass arguments to satisfy mypy strict typing
        results = collection.query(
            query_embeddings=cast(Any, query_embeddings),
            n_results=n_results,
            where=where,
            where_document=cast(Any, where_document),
        )

        # Transform to standard SearchResult objects
        output = []

        # Chroma returns lists of lists (batch queries).
        # We only support batch=1 for now based on usage.

        # Safe access with explicit type checks for mypy
        ids = results.get("ids")
        if not ids or not ids[0]:
            return []

        # We know we have at least one list of IDs
        batch_ids = ids[0]
        num_results = len(batch_ids)

        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")

        for i in range(num_results):
            _id = batch_ids[i]

            # Safe indexing: check if the outer list and the inner list exist
            _content = ""
            if documents and len(documents) > 0 and documents[0] is not None:
                _content = documents[0][i]

            _meta: Dict[str, Any] = {}
            if metadatas and len(metadatas) > 0 and metadatas[0] is not None:
                # cast to dict[str, Any] as chroma metadata is closer to that
                _meta = cast(Dict[str, Any], metadatas[0][i])

            _dist = 0.0
            if distances and len(distances) > 0 and distances[0] is not None:
                _dist = distances[0][i]

            output.append(
                SearchResult(id=_id, content=_content, metadata=_meta, distance=_dist)
            )

        return output

    def get(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        collection = self._client.get_collection(name=collection_name)
        kwargs: Dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        if include:
            kwargs["include"] = include

        return cast(Dict[str, Any], collection.get(**kwargs))

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        collection = self._client.get_collection(name=collection_name)
        kwargs: Dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where

        collection.delete(**kwargs)
