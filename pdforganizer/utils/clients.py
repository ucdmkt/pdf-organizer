"""Client initialization patterns for GenAI and ChromaDB."""

from google import genai

from pdforganizer import config


def get_genai_client():
    """Initializes and returns the GenAI client."""
    if config.SETTINGS.google_cloud_project:
        return genai.Client(
            vertexai=True,
            project=config.SETTINGS.google_cloud_project,
            location=config.SETTINGS.google_cloud_location,
        )
    return genai.Client(api_key=config.SETTINGS.google_api_key)
