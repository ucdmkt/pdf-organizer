"""Google GenAI interaction utilities for OCR and Embedding."""

import concurrent.futures
import mimetypes
import re
import time

from google.genai import types

from pdforganizer import config
from pdforganizer.utils.resilience import retry_with_backoff

OCR_PROMPT = (
    "Perform full OCR on the given file, and the OCRed text in Markdown format with "
    "tables, headers and image placeholders, preserving logical layout and formatting "
    "as much as possible. Generate only the Markdown text, **DO NOT wrap it in any "
    "text or code block markers, such as ```markdown ... ```: "
    "Strictly only generate OCRed contents.**"
)


def _upload_file_ai_studio(client, path_obj, mime_type, logger):
    """
    Handles file upload for AI Studio (File API).
    """
    with open(path_obj, "rb") as f:
        return retry_with_backoff(logger)(client.files.upload)(
            file=f,
            config=types.UploadFileConfig(
                mime_type=mime_type, display_name=path_obj.name
            ),
        )


def upload_file(client, path_obj, logger, doc_id="unknown", rel_path="unknown"):
    """
    Uploads a file for GenAI processing.
    Delegates to appropriate backend strategy (Vertex vs AI Studio).
    """
    try:
        mime_type, _ = mimetypes.guess_type(path_obj)
        if not mime_type:
            mime_type = "application/pdf"

        return _upload_file_ai_studio(client, path_obj, mime_type, logger)
    except Exception:
        logger.error(
            "❌ Failed to upload document",
            exc_info=True,
            extra={"doc_id": doc_id, "rel_path": rel_path},
        )
        return None


def ocr_document(client, path_obj, logger, doc_id="unknown", rel_path="unknown"):
    """
    Uploads a file and performs OCR using the configured model.
    Returns the extracted text in Markdown format or None on failure.
    """
    try:
        logger.info(
            "📸 Multi-page OCR Scan", extra={"doc_id": doc_id, "rel_path": rel_path}
        )

        uploaded_file = upload_file(
            client, path_obj, logger, doc_id=doc_id, rel_path=rel_path
        )
        if not uploaded_file:
            return None

        try:
            time.sleep(1)  # Allow for processing on the server

            response = retry_with_backoff(logger)(client.models.generate_content)(
                **build_ocr_request(uploaded_file)
            )

            text = response.text
            if text:
                text = clean_ocr_text(text)

            return text
        finally:
            pass

    except Exception:
        logger.error(
            "❌ Failed to process document",
            exc_info=True,
            extra={"doc_id": doc_id, "rel_path": rel_path},
        )
        return None


def clean_ocr_text(text):
    """
    Cleans the OCR text by removing markdown code block markers.
    """
    if not text:
        return text

    text = text.strip()

    # Loop to remove multiple layers of backticks if present
    while True:
        original = text
        # Remove starting block
        text = re.sub(r"^```[\w-]*\s*", "", text, flags=re.IGNORECASE).strip()
        # Remove ending block
        text = re.sub(r"\s*```$", "", text).strip()

        if text == original:
            break

    return text


def build_ocr_request(uploaded_file):
    """
    Constructs the standard OCR request body.
    Ensures consistent prompting across interactive and batch modes.
    """
    return {
        "model": config.SETTINGS.ocr_model_id,
        "contents": [uploaded_file, OCR_PROMPT],
    }


def build_batch_ocr_request(uploaded_file, doc_id):
    """
    Constructs the batch request entry for OCR.
    Returns (jsonl_line_dict, file_uri) or (None, None) on error.
    """
    if not uploaded_file:
        return None, None

    uri = getattr(uploaded_file, "uri", None)
    if not uri:
        return None, None

    file_part = {
        "file_data": {
            "file_uri": uri,
            "mime_type": getattr(uploaded_file, "mime_type", "application/pdf"),
        }
    }

    text_part = {"text": OCR_PROMPT}

    # Construct strict request payload
    req_clean = {
        "contents": [{"role": "user", "parts": [file_part, text_part]}],
    }

    jsonl_line = {"custom_id": doc_id, "request": req_clean}
    return jsonl_line, uri


def get_embedding_params(task_type="RETRIEVAL_DOCUMENT"):
    """
    Returns the shared configuration parameters for embedding.
    Useful for ensuring consistency between online and batch modes.
    """
    return {
        "task_type": task_type,
        "output_dimensionality": config.SETTINGS.embed_dimension,
        "title": "Document chunk" if task_type == "RETRIEVAL_DOCUMENT" else None,
    }


def build_batch_embedding_request(text, task_type="RETRIEVAL_DOCUMENT"):
    """
    Constructs the full request dictionary for a batch embedding job.
    Encapsulates model, content structure, and config params.
    """
    request = {
        "model": config.SETTINGS.embed_model_id,
        "content": {"parts": [{"text": text}]},
    }
    request.update(get_embedding_params(task_type))
    return request


def embed_text(client, text, logger, task_type="RETRIEVAL_DOCUMENT"):
    """
    Embeds text using the configured embedding model.
    Returns the embedding response.
    """
    params = get_embedding_params(task_type)
    return retry_with_backoff(logger)(client.models.embed_content)(
        model=config.SETTINGS.embed_model_id,
        contents=text,
        config=types.EmbedContentConfig(**params),
    )


def batch_upload_file(client, path_objs, doc_ids, logger, max_workers=10):
    """
    Parallelizes file uploads using ThreadPoolExecutor.
    Returns a list of uploaded file objects (preserving order).
    Some entries may be None if individual uploads fail.
    """
    logger.info(
        f"🚀 Starting parallel upload of {len(path_objs)} files "
        f"(max_workers={max_workers})..."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map preserves the order of the input iterables
        return list(
            executor.map(
                lambda p: upload_file(client, p[0], logger, doc_id=p[1]),
                zip(path_objs, doc_ids),
            )
        )
