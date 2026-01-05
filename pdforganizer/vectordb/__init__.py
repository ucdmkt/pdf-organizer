"""Vector Database Abstraction Layer."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pdforganizer import config

LOGGER = logging.getLogger(__name__)

__all__ = ["VectorDBClient", "SearchResult", "get_vector_db"]


@dataclass
class SearchResult:
    """
    Standardized return object for vector searches.

    Attributes:
        id (str): The unique identifier of the document (usually the content hash).
        content (str): The text content of the document chunk.
        metadata (Dict[str, Any]): Associated metadata (e.g., filename, relative path).
        distance (float): The distance score from the query embedding
                          (lower is usually better/closer).
    """

    id: str
    content: str
    metadata: Dict[str, Any]
    distance: float


class VectorDBClient(ABC):
    """Abstract interface for Vector Database interactions."""

    @abstractmethod
    def get_or_create_collection(self, name: str) -> Any:
        """Initialize the collection."""
        pass

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update records."""
        pass

    @abstractmethod
    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Query the database.
        Returns a list of SearchResult objects.
        """
        pass

    @abstractmethod
    def get(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve records by ID or filter."""
        pass

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete records by ID or filter."""
        pass


def get_vector_db() -> VectorDBClient:
    """
    Returns the configured VectorDB client.

    Currently hardcoded to ChromaDBClient (local) to maintain
    strict backward compatibility and internal refactor requirements.
    In the future, this can read `config.SETTINGS.vectordb_provider`.
    """
    # Import here to avoid circular dependency
    # vectordb -> chroma_client -> vectordb (for base class)
    from pdforganizer.vectordb.chroma_client import ChromaDBClient

    # Simply return ChromaDBClient with current global settings
    return ChromaDBClient(config.SETTINGS)
