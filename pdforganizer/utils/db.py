import logging


def save_to_vector_store(
    collection, doc_id, full_path, rel_path, content, values, logger=None
):
    """
    Shared logic for logging and saving to ChromaDB.
    Ensures consistency between batch indexer and interactive analyzer.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(
        "💾 Saving to ChromaDB",
        extra={
            "doc_id": doc_id,
            "record": {
                "ids": [doc_id],
                # Convert Path objects to string if necessary,
                # though Chroma handles strings usually
                "metadatas": [
                    {"filename": str(full_path.name), "rel_path": str(rel_path)}
                ],
                "documents": [content[:100]],
            },
        },
    )

    collection.upsert(
        ids=[doc_id],
        embeddings=[values],
        metadatas=[{"filename": str(full_path.name), "rel_path": str(rel_path)}],
        documents=[content],
    )
    logger.info("✅ Indexed successfully.", extra={"doc_id": doc_id})
