import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Helper to ensure we can import the package
sys.path.append(str(Path.cwd()))


@pytest.fixture
def mock_genai_client(mocker):
    """Mocks the Google GenAI Client."""
    mock_client = MagicMock()
    # Mocking models.generate_content
    mock_client.models.generate_content.return_value = MagicMock(
        text="Mocked OCR Content", parsed=None
    )
    # Mocking models.embed_content
    mock_client.models.embed_content.return_value = MagicMock(
        embeddings=[MagicMock(values=[0.1] * 768)]
    )
    # Mocking files.upload
    mock_client.files.upload.return_value = MagicMock(
        uri="gs://mock/file", name="files/mock", mime_type="application/pdf"
    )

    mocker.patch("pdforganizer.utils.clients.genai.Client", return_value=mock_client)
    mocker.patch("pdforganizer.utils.get_genai_client", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_chroma_client(mocker):
    """Mocks the ChromaDB Client and Collection."""
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mocker.patch(
        "pdforganizer.utils.clients.chromadb.PersistentClient", return_value=mock_client
    )
    mocker.patch("pdforganizer.utils.get_chroma_client", return_value=mock_client)
    mocker.patch("pdforganizer.utils.get_collection", return_value=mock_collection)

    return mock_client, mock_collection
