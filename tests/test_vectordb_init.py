"""Unit tests for VectorDB package initialization."""

import unittest
from unittest.mock import patch

from pdforganizer import vectordb


class TestVectorDBInit(unittest.TestCase):
    @patch("pdforganizer.vectordb.config")
    @patch("pdforganizer.vectordb.chroma_client.ChromaDBClient")
    def test_get_vector_db_returns_chroma(self, mock_chroma_cls, mock_config):
        """Test that get_vector_db currently returns ChromaDBClient."""
        client = vectordb.get_vector_db()
        mock_chroma_cls.assert_called_once()
        self.assertTrue(bool(client))
