"""Unit tests for ChromaDBClient adapter."""

import unittest
from unittest.mock import MagicMock, patch

from pdforganizer.config import AppConfig
from pdforganizer.vectordb.chroma_client import ChromaDBClient


class TestChromaDBClient(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock(spec=AppConfig)
        self.mock_config.db_path = "/tmp/test_db"
        self.mock_config.collection_name = "test_collection"

    @patch("pdforganizer.vectordb.chroma_client.chromadb.PersistentClient")
    def test_init_local(self, mock_persistent):
        """Test initialization of local client."""
        client = ChromaDBClient(self.mock_config)
        mock_persistent.assert_called_once_with(path="/tmp/test_db")
        self.assertIsNotNone(client)

    @patch("pdforganizer.vectordb.chroma_client.chromadb.PersistentClient")
    def test_upsert(self, mock_persistent):
        """Test upsert delegation."""
        mock_client_instance = mock_persistent.return_value
        mock_collection = mock_client_instance.get_collection.return_value

        client = ChromaDBClient(self.mock_config)
        client.upsert(
            collection_name="test_collection",
            ids=["id1"],
            documents=["doc1"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"meta": "data"}],
        )

        mock_client_instance.get_collection.assert_called_with(name="test_collection")
        mock_collection.upsert.assert_called_once_with(
            ids=["id1"],
            documents=["doc1"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"meta": "data"}],
        )

    @patch("pdforganizer.vectordb.chroma_client.chromadb.PersistentClient")
    def test_query_format(self, mock_persistent):
        """Test query result formatting to SearchResult objects."""
        mock_client_instance = mock_persistent.return_value
        mock_collection = mock_client_instance.get_collection.return_value

        # Mock Chroma return format
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"path": "foo"}]],
            "distances": [[0.5]],
        }

        client = ChromaDBClient(self.mock_config)
        results = client.query(
            collection_name="test_collection", query_embeddings=[[0.1]], n_results=1
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "id1")
        self.assertEqual(results[0].content, "doc1")
        self.assertEqual(results[0].metadata, {"path": "foo"})
        self.assertEqual(results[0].distance, 0.5)

    @patch("pdforganizer.vectordb.chroma_client.chromadb.PersistentClient")
    def test_delete(self, mock_persistent):
        """Test delete delegation."""
        mock_client_instance = mock_persistent.return_value
        mock_collection = mock_client_instance.get_collection.return_value

        client = ChromaDBClient(self.mock_config)
        client.delete(collection_name="test_collection", ids=["id1"])

        mock_collection.delete.assert_called_once_with(ids=["id1"])
