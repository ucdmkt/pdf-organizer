"""Batch indexing module for the PDF Organizer."""

import argparse
import fnmatch
import hashlib
import json
import tempfile
import time
from pathlib import Path

from google.genai import types

from pdforganizer import config, utils
from pdforganizer.vectordb import get_vector_db

LOGGER = utils.setup_logger(__name__)


def _is_blocklisted(doc_id):
    """Checks if a document ID matches any glob pattern in the blocklist."""
    return any(
        fnmatch.fnmatch(doc_id, pattern) for pattern in config.SETTINGS.index_blocklist
    )


def _init_services():
    client = utils.get_genai_client()
    vector_db = get_vector_db()
    vector_db.get_or_create_collection(config.SETTINGS.collection_name)
    return client, vector_db


def _poll_job(client, job_name, interval=10):
    """
    Polls a batch job until it succeeds or fails.
    Returns the job object on success.
    Raises RuntimeError on failure.
    """
    LOGGER.info(f"⏳ Monitoring job: {job_name}")
    while True:
        job = utils.retry_with_backoff(LOGGER)(client.batches.get)(name=job_name)
        state = job.state
        if state == types.JobState.JOB_STATE_SUCCEEDED:
            return job
        elif state in [
            types.JobState.JOB_STATE_FAILED,
            types.JobState.JOB_STATE_CANCELLED,
        ]:
            raise RuntimeError(f"Batch Job {job_name} failed with state: {state}")
        LOGGER.info(f"⏳ Waiting for job {job_name}... Status: {state}")
        time.sleep(interval)


def _scan_and_sync_files(docs_base_path, vector_db, seen_ids_accumulator):
    """
    Scans files, syncs metadata for moved files, and yields new files for indexing.
    Populates seen_ids_accumulator in place with valid file content hashes.
    """
    docs_base_path = Path(docs_base_path).expanduser().resolve()
    if not docs_base_path.exists() or not docs_base_path.is_dir():
        LOGGER.error(
            f"Documents base directory not found or not a directory: {docs_base_path}"
        )
        return

    LOGGER.info(f"📂 Scanning directory: {docs_base_path}")

    # Recursive scan for all PDF files
    for full_path_obj in docs_base_path.rglob("*.pdf"):
        if not full_path_obj.is_file():
            continue

        # Calculate relative path string
        try:
            rel_path = str(full_path_obj.relative_to(docs_base_path))
        except ValueError:
            rel_path = full_path_obj.name

        # Blocklist check
        if _is_blocklisted(rel_path):
            continue

        # 1. Compute Content Hash
        try:
            file_hash = utils.generic.compute_file_hash(full_path_obj)
            doc_id = f"file_content_hash:{file_hash}"
            seen_ids_accumulator.add(doc_id)

            # 2. Check DB
            existing = vector_db.get(
                collection_name=config.SETTINGS.collection_name,
                ids=[doc_id],
            )
            if existing and existing.get("ids"):
                # File exists in DB (Hash Match)
                # Check if path metadata needs update (Move detection)
                existing_meta = existing.get("metadatas")[0]
                existing_rel_path = existing_meta.get("rel_path")

                if existing_rel_path != rel_path:
                    LOGGER.info(
                        f"🔄 File moved: {existing_rel_path} -> {rel_path}. "
                        "Updating metadata."
                    )
                    existing_meta["rel_path"] = rel_path
                    existing_meta["filename"] = full_path_obj.name
                    vector_db.update(
                        collection_name=config.SETTINGS.collection_name,
                        ids=[doc_id],
                        metadatas=[existing_meta],
                    )
                else:
                    LOGGER.info(
                        "⏭️ Skipped (already indexed)",
                        extra={"doc_id": doc_id},
                    )

                continue

            # 3. New Content -> Yield for indexing
            yield doc_id, full_path_obj

        except Exception as e:
            LOGGER.error(f"❌ Failed to process file {rel_path}: {e}")


def _compute_config_hash():
    """Computes MD5 hash of the entire AppConfig."""
    # Use model_dump to get a dict, then json dumps with sort_keys for stability
    # mode='json' converts Paths to strings etc.
    config_dict = config.SETTINGS.model_dump(mode="json")

    data_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(data_str.encode("utf-8")).hexdigest()


def _save_batch_state(state, merge=False):
    """Saves the current batch state to a predictable file location."""
    config.SETTINGS.batch_state_file.parent.mkdir(parents=True, exist_ok=True)

    final_state = state
    if merge and config.SETTINGS.batch_state_file.exists():
        try:
            existing = _load_batch_state()
            if existing:
                final_state = existing.copy()
                final_state.update(state)
        except Exception:
            pass

    # Always inject current config hash
    final_state["config_hash"] = _compute_config_hash()

    with open(config.SETTINGS.batch_state_file, "w", encoding="utf-8") as f:
        # Convert Path objects to strings for JSON serialization
        serializable_metadata = [
            [m[0], str(m[1])] for m in final_state.get("metadata", [])
        ]
        state_copy = final_state.copy()
        state_copy["metadata"] = serializable_metadata
        json.dump(state_copy, f, indent=2)


def _load_batch_state():
    """Loads the batch state if it exists and configuration matches."""
    if config.SETTINGS.batch_state_file.exists():
        try:
            with open(config.SETTINGS.batch_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

                # Check Config Hash
                current_hash = _compute_config_hash()
                saved_hash = state.get("config_hash")

                if saved_hash != current_hash:
                    LOGGER.warning(
                        "⚠️ Config changed since last run. "
                        "Invalidating saved batch state."
                    )
                    _clear_batch_state()
                    return None

                # Convert strings back to Path objects
                state["metadata"] = [
                    [m[0], Path(m[1])] for m in state.get("metadata", [])
                ]
                return state
        except Exception as e:
            LOGGER.error(f"❌ Failed to load state file: {e}")
            return None
    return None


def _clear_batch_state():
    """Clears the batch state file."""
    if config.SETTINGS.batch_state_file.exists():
        config.SETTINGS.batch_state_file.unlink()
        LOGGER.info("🧹 Batch state cleared.")


def _run_auto_purge(vector_db, seen_ids):
    """
    Removes records from VectorDB that were not seen in the current scan (Orphans).
    """
    LOGGER.info("🧹 Starting database auto-purge check...")

    # Retrieve all IDs from the collection
    res = vector_db.get(collection_name=config.SETTINGS.collection_name)
    all_ids = set(res.get("ids", []))

    orphans = all_ids - seen_ids

    if orphans:
        LOGGER.info(
            f"⚠️ Found {len(orphans)} orphan records "
            "(missing or blocklisted files). Purging..."
        )
        vector_db.delete(
            collection_name=config.SETTINGS.collection_name, ids=list(orphans)
        )
        LOGGER.info("✅ Purge complete.")
    else:
        LOGGER.info("✅ Database is clean (no orphans).")


def _manage_ocr_phase(client, chunk_files):
    """Initiates OCR batch job via JSONL file to ensure ID mapping."""
    doc_ids = [cf[0] for cf in chunk_files]
    path_objs = [cf[1] for cf in chunk_files]

    # Calculate rel_paths for logging context (using path objects directly)

    # 1. Parallel Upload
    uploaded_files = utils.genai.batch_upload_file(client, path_objs, doc_ids, LOGGER)

    # Collect lines and metadata
    jsonl_lines = []
    batch_metadata = []

    for doc_id, full_path_obj, uploaded_file in zip(doc_ids, path_objs, uploaded_files):
        jsonl_line, _ = utils.genai.build_batch_ocr_request(uploaded_file, doc_id)

        if not jsonl_line:
            LOGGER.error(
                f"❌ Could not build batch request for {doc_id} "
                "(Upload failed or invalid mode)",
                extra={"doc_id": doc_id},
            )
            continue

        jsonl_lines.append(json.dumps(jsonl_line))
        batch_metadata.append((doc_id, full_path_obj))

    if not jsonl_lines:
        return None, None

    LOGGER.info(f"🚀 Starting OCR Batch Job for {len(jsonl_lines)} items...")

    # Create temp JSONL
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=True, encoding="utf-8"
    ) as tmp_f:
        tmp_f.write("\n".join(jsonl_lines))
        tmp_f.flush()

        upload_file = utils.retry_with_backoff(LOGGER)(client.files.upload)(
            file=Path(tmp_f.name), config={"mime_type": "application/json"}
        )
        upload_ref = types.BatchJobSource(file_name=upload_file.name)

    batch_job = utils.retry_with_backoff(LOGGER)(client.batches.create)(
        model=config.SETTINGS.ocr_model_id,
        src=upload_ref,
    )
    job_id = batch_job.name
    job_type = "OCR"

    _save_batch_state(
        {"job_id": job_id, "job_type": job_type, "metadata": batch_metadata}
    )

    return job_id, batch_metadata


def _wait_for_ocr_and_extract(client, job_id, batch_metadata):
    """Waits for OCR job and returns valid texts mapped by ID."""
    batch_job = _poll_job(client, job_id, interval=10)
    LOGGER.info("✅ OCR Batch Job Completed! Processing results...")

    # Create lookup map for metadata
    meta_map = {doc_id: path for doc_id, path in batch_metadata}

    valid_results_map = []

    try:
        output_file_name = batch_job.dest.file_name
        results_list = []

        if output_file_name:
            content_resp = client.files.download(file=output_file_name)
            jsonl_content = content_resp.decode("utf-8")
            results_list = [
                json.loads(line) for line in jsonl_content.splitlines() if line.strip()
            ]
        elif (
            hasattr(batch_job.dest, "inlined_responses")
            and batch_job.dest.inlined_responses
        ):
            # Inlined responses usually don't support custom_id as cleanly
            # in the Python object
            # unless we access a specific field. Assuming standard batch usage via file.
            LOGGER.warning(
                "⚠️ unexpected inlined_responses for batch job, "
                "falling back to index assumption (risky)"
            )
            results_list = [
                item.response.model_dump(mode="json")
                for item in batch_job.dest.inlined_responses
            ]

        processed_ids = set()

        for result in results_list:
            # Handle JSONL format: {"custom_id": "...", "response": {...}}
            custom_id = result.get("custom_id")

            # Fallback for inlined or malformed
            if not custom_id:
                # If we absolutely cannot find an ID, we skip to avoid corruption
                LOGGER.error(
                    "❌ Result missing custom_id, skipping to prevent corruption."
                )
                continue

            if custom_id not in meta_map:
                LOGGER.warning(
                    f"⚠️ Unknown custom_id {custom_id} in results, " "ignoring."
                )
                continue

            full_path_obj = meta_map[custom_id]
            processed_ids.add(custom_id)

            try:
                actual_res = result.get("response", result)
                # Check for error status in response
                if "error" in actual_res:
                    LOGGER.error(
                        f"❌ Batch Item Error for {custom_id}: {actual_res['error']}"
                    )
                    continue

                candidates = actual_res.get("candidates", [{}])
                if not candidates:
                    LOGGER.warning(f"⚠️ No candidates for {custom_id}")
                    continue

                content = (
                    candidates[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                content = utils.genai.clean_ocr_text(content)

                if content:
                    valid_results_map.append(
                        {
                            "doc_id": custom_id,
                            "full_path": full_path_obj,
                            "content": content,
                        }
                    )
            except Exception as e:
                LOGGER.error(f"❌ Error parsing OCR result for {custom_id}: {e}")

        # Check for missing files
        missing = set(meta_map.keys()) - processed_ids
        if missing:
            LOGGER.warning(
                f"⚠️ {len(missing)} documents missing from batch results: {missing}"
            )

        return valid_results_map

    except Exception as e:
        LOGGER.error(f"❌ Failed to process OCR results: {e}", exc_info=True)
        return []
    finally:
        pass


def _manage_embedding_phase(client, valid_map, batch_metadata):
    """Initiates embedding batch job using custom_id for correlation."""
    LOGGER.info(f"🚀 Starting Embedding Batch Job for {len(valid_map)} items...")

    jsonl_lines = []

    for item in valid_map:
        doc_id = item["doc_id"]
        text = item["content"]
        req = utils.build_batch_embedding_request(text)
        # Use custom_id to ensure we can map back results correctly
        line = {"custom_id": doc_id, "request": req}
        jsonl_lines.append(json.dumps(line))

    try:
        # AI Studio Mode: Use File API with create_embeddings helper
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=True, encoding="utf-8"
        ) as tmp_f:
            tmp_f.write("\n".join(jsonl_lines))
            tmp_f.flush()
            upload_ref = utils.retry_with_backoff(LOGGER)(client.files.upload)(
                file=Path(tmp_f.name), config={"mime_type": "application/json"}
            )
        src = types.EmbeddingsBatchJobSource(file_name=upload_ref.name)

        embed_job = utils.retry_with_backoff(LOGGER)(client.batches.create_embeddings)(
            model=config.SETTINGS.embed_model_id,
            src=src,
        )

        job_id = embed_job.name

        # Save valid_map related info so we can reconstruct it on resume
        # actually we save 'texts' and 'metadata' in older logic.
        # To be compatible with resume structure:
        texts = [i["content"] for i in valid_map]

        _save_batch_state(
            {
                "job_id": job_id,
                "job_type": "EMBEDDING",
                "metadata": batch_metadata,
                "texts": texts,
            }
        )
        return job_id
    except Exception as e:
        LOGGER.error(f"❌ Embedding job creation failed: {e}", exc_info=True)
        return None


def _wait_for_embedding_and_index(
    client, vector_db, job_id, batch_metadata, valid_map, docs_base_path
):
    """Waits for Embedding job and finalizes indexing using ID mapping."""
    embed_job = _poll_job(client, job_id, interval=5)

    LOGGER.info("✅ Embedding Batch Job Completed! Finalizing indexing...")

    # Create lookup map
    item_map = {item["doc_id"]: item for item in valid_map}

    try:
        emb_output_name = embed_job.dest.file_name
        emb_results_list = []

        if emb_output_name:
            emb_content_resp = client.files.download(file=emb_output_name)
            emb_jsonl = emb_content_resp.decode("utf-8")
            emb_results_list = [
                json.loads(line) for line in emb_jsonl.splitlines() if line.strip()
            ]
        elif (
            hasattr(embed_job.dest, "inlined_responses")
            and embed_job.dest.inlined_responses
        ):
            # Fallback index-based if allowed or assume standard format
            # Usually standard batch gives custom_id.
            emb_results_list = [
                item.response.model_dump(mode="json")
                for item in embed_job.dest.inlined_responses
            ]

        processed_ids = set()

        for result in emb_results_list:
            custom_id = result.get("custom_id")
            if not custom_id:
                # Fallback to key if present (legacy)
                custom_id = result.get("key")

            if not custom_id:
                LOGGER.error("❌ Embedding result missing custom_id/key, skipping.")
                continue

            if custom_id not in item_map:
                LOGGER.warning(f"⚠️ Unknown embedding result ID {custom_id}")
                continue

            processed_ids.add(custom_id)
            item = item_map[custom_id]

            emb_result = result.get("response", result)
            values = emb_result.get("embedding", {}).get("values") or emb_result.get(
                "values"
            )

            if values:
                try:
                    rel_path = str(item["full_path"].relative_to(docs_base_path))
                except ValueError:
                    rel_path = item["full_path"].name

                utils.db.save_to_vector_store(
                    vector_db,
                    config.SETTINGS.collection_name,
                    item["doc_id"],
                    item["full_path"],
                    rel_path,
                    item["content"],
                    values,
                    LOGGER,
                )

        # Check missing
        missing = set(item_map.keys()) - processed_ids
        if missing:
            LOGGER.warning(f"⚠️ {len(missing)} items failed to embed: {missing}")

        _clear_batch_state()
        LOGGER.info("✅ Chunk processing complete.")
    except Exception as e:
        LOGGER.error(f"❌ Failed to finalize indexing: {e}", exc_info=True)


def _run_batch(client, vector_db, docs_base_path, batch_size=100):

    def process_new_chunk(chunk_files):
        """Standard flow for processing a fresh chunk of files."""
        # 1. Start OCR
        job_id, batch_metadata = _manage_ocr_phase(client, chunk_files)
        if not job_id:
            return

        # 2. Start Embedding (Wait for OCR first)
        valid_map = _wait_for_ocr_and_extract(client, job_id, batch_metadata)
        if not valid_map:
            _clear_batch_state()
            return

        emb_job_id = _manage_embedding_phase(client, valid_map, batch_metadata)
        if not emb_job_id:
            return

        # 3. Finalize
        _wait_for_embedding_and_index(
            client, vector_db, emb_job_id, batch_metadata, valid_map, docs_base_path
        )

    def resume_existing_job(saved_state):
        """Resume flow for an interrupted job."""
        job_id = saved_state.get("job_id")
        job_type = saved_state.get("job_type")
        batch_metadata = saved_state.get("metadata", [])
        texts = saved_state.get("texts", [])

        LOGGER.info(f"🔄 Resuming {job_type} job: {job_id}")

        if job_type == "OCR":
            # Resume waiting for OCR
            valid_map = _wait_for_ocr_and_extract(client, job_id, batch_metadata)
            if not valid_map:
                _clear_batch_state()
                return

            # Start Embedding
            emb_job_id = _manage_embedding_phase(client, valid_map, batch_metadata)
            if not emb_job_id:
                return

            # Finalize
            # Note: valid_map is fresh here
            _wait_for_embedding_and_index(
                client,
                vector_db,
                emb_job_id,
                batch_metadata,
                valid_map,
                docs_base_path,
            )

        elif job_type == "EMBEDDING":
            # Reconstruct valid_map from saved texts + metadata first
            valid_map = []
            for i in range(len(batch_metadata)):
                valid_map.append(
                    {
                        "doc_id": batch_metadata[i][0],
                        "full_path": batch_metadata[i][1],
                        "content": utils.genai.clean_ocr_text(texts[i]),
                    }
                )

            # If job_id is None, it means we crashed *before* creating the embedding job
            # but *after* saving the OCR results. so we need to Create it now.
            if not job_id:
                job_id = _manage_embedding_phase(client, valid_map, batch_metadata)
                if not job_id:
                    return

            # Resume waiting for Embedding
            _wait_for_embedding_and_index(
                client, vector_db, job_id, batch_metadata, valid_map, docs_base_path
            )

    # 1. Startup Recovery Check
    saved_state = _load_batch_state()
    if saved_state:
        resume_existing_job(saved_state)

    # 2. Main Batch Loop
    current_chunk = []
    seen_ids = set()

    for doc_id, full_path_obj in _scan_and_sync_files(
        docs_base_path, vector_db, seen_ids
    ):
        current_chunk.append((doc_id, full_path_obj))
        if len(current_chunk) >= batch_size:
            process_new_chunk(current_chunk)
            current_chunk = []

    if current_chunk:
        process_new_chunk(current_chunk)

    # Auto-Purge at the end of the run
    _run_auto_purge(vector_db, seen_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index PDF documents using Gemini and ChromaDB."
    )
    parser.add_argument(
        "--docs-base-dir",
        type=str,
        default=str(config.SETTINGS.docs_base_dir),
        help=(
            "Path to the base directory containing documents to index "
            "(recursive search enabled)."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files to process per batch chunk (default: 100).",
    )
    args = parser.parse_args()

    # Initialize services once
    client, vector_db = _init_services()

    _run_batch(client, vector_db, args.docs_base_dir, batch_size=args.batch_size)
