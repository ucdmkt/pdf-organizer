"""ChromaDB implementation of VectorDBClient."""

import logging
from typing import Any, Dict, List, Optional

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
            embeddings=embeddings,
            metadatas=metadatas,
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

        # Chroma query args
        args = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
        }
        if where:
            args["where"] = where
        if where_document:
            args["where_document"] = where_document

        results = collection.query(**args)

        # Transform to standard SearchResult objects
        output = []

        # Chroma returns lists of lists (batch queries).
        # We only support batch=1 for now based on usage.
        # But let's handle the structure safely.
        # results['ids'] is [[id1, id2...]]

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        # Iterate over the first batch result
        num_results = len(results["ids"][0])
        for i in range(num_results):
            _id = results["ids"][0][i]
            _content = results["documents"][0][i] if results.get("documents") else ""
            _meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            _dist = results["distances"][0][i] if results.get("distances") else 0.0

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

        return collection.get(**kwargs)

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
