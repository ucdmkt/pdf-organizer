"""Client initialization patterns for GenAI and ChromaDB."""

import chromadb
from google import genai

from pdforganizer import config


def get_genai_client():
    """Initializes and returns the GenAI client."""
    return genai.Client(api_key=config.settings.google_api_key)


def get_chroma_client():
    """Initializes and returns the ChromaDB persistent client."""
    return chromadb.PersistentClient(path=str(config.settings.db_path))


def get_collection(client):
    """Gets or creates the ChromaDB collection."""
    return client.get_or_create_collection(name=config.settings.collection_name)
