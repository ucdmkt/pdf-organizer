"""Analyzer module for identifying document similarities and suggesting organization."""

import argparse
import fnmatch
import json
import shutil
import time
from pathlib import Path

from google.genai import types
from pydantic import BaseModel, Field

from pdforganizer import config, utils
from pdforganizer.vectordb import get_vector_db

PROMPT_TEMPLATE = (
    "I have a new document to organize. "
    "Here are {n_neighbors} similar valid documents "
    "and their paths and filenames from my library:\n\n"
    "{history}\n\n"
    "Based on these examples, suggest a folder path and filename "
    "for the new document attached.\n"
    "* You need to follow the same naming convention of the filename "
    "as the provided examples based on the content of the document. .\n"
    "* If you do not find suitable folder or sub folder in the historical examples, "
    "do not propose any new folder or sub folder unless it is absolutely suitable "
    "based on a pattern in the provided historical examples. "
    "In such a case, propose no folder or sub folder, but keep it empty.\n\n"
    "The new document in markdown is shown below:\n"
)

LOGGER = utils.setup_logger(__name__)  # Initialize LOGGER at module level
_AUTO_APPLY_CONFIDENCE_THRESHOLD = 0.94


class _FileMoveSuggestion(BaseModel):
    target_folder: str = Field(
        description=(
            "The existing folder path where the document should be moved. "
            "Keep empty if no suitable folder exists."
        )
    )
    new_filename: str = Field(
        description=(
            "The new filename for the document, following the "
            "naming convention of existing similar documents."
        )
    )
    reasoning: str = Field(
        description=(
            "The reasoning behind the suggested folder and filename. "
            "If confidence score is less than 100%, explain why could make "
            "confidence higher."
        )
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1.")


def _cli_output(msg, level="info", log_only=False, **kwargs):
    """Consolidated helper for CLI print and structured logging."""
    if not log_only:
        print(msg, flush=True)
    if level == "error":
        LOGGER.error(msg, **kwargs)
    else:
        LOGGER.info(msg, **kwargs)


def _perform_move(abs_source_path, dest_path):
    """Safely moves source to dest, handling directory creation and existing files."""
    try:
        # Create parents if missing
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing destination
        if dest_path.exists():
            if abs_source_path == dest_path:
                _cli_output("ℹ️  File is already at the destination. Skipping.")
            else:
                _cli_output(f"⚠️  Overwriting existing file: {dest_path}")
                dest_path.unlink()
                shutil.move(str(abs_source_path), str(dest_path))
                _cli_output(f"✅ Successfully moved to {dest_path}")
        else:
            # Execute move
            shutil.move(str(abs_source_path), str(dest_path))
            _cli_output(f"✅ Successfully moved to {dest_path}")
        return True
    except Exception as e:
        _cli_output(f"❌ Failed to move file: {e}", level="error")
        return False


def analyze_file(rel_file_path, docs_base_dir):
    """
    Analyzes a file using GenAI and returns the move suggestion and
    absolute source path.
    """
    client = utils.get_genai_client()
    vector_db = get_vector_db()
    # Ensure collection exists (though get/query might auto-lazy load in implementation)
    vector_db.get_or_create_collection(config.SETTINGS.collection_name)

    # Enforce resolution relative to provided base directory
    # Enforce resolution relative to provided base directory
    path_obj = (Path(docs_base_dir) / rel_file_path).resolve()

    if not path_obj.exists():
        return f"❌ File not found: {path_obj}"

    # Use the absolute path from Path.resolve() so we're working
    # with the true destination on disk.
    abs_source_path = path_obj

    # 0. Check if file content is already in DB
    file_hash = utils.generic.compute_file_hash(abs_source_path)
    # Use hash-based ID for consistency in logging
    doc_id = f"file_content_hash:{file_hash}"

    LOGGER.info(
        "🔎 Analyzing document", extra={"doc_id": doc_id, "rel_path": rel_file_path}
    )

    LOGGER.info(
        "🔍 Checking DB for existing hash: {doc_id}",
        extra={"doc_id": doc_id, "rel_path": rel_file_path},
    )
    existing_record = vector_db.get(
        collection_name=config.SETTINGS.collection_name,
        ids=[doc_id],
        include=["documents", "metadatas", "embeddings"],
    )

    doc_markdown = None
    embedding_values = None

    if existing_record and existing_record["ids"]:
        LOGGER.info(
            "♻️  Found existing record. Reusing content and embeddings.",
            extra={"doc_id": doc_id, "rel_path": rel_file_path},
        )
        doc_markdown = existing_record["documents"][0]
        if (
            existing_record.get("embeddings") is not None
            and len(existing_record["embeddings"]) > 0
        ):
            embedding_values = existing_record["embeddings"][0]

    # 1. OCR scan of new file (if not found in DB)
    if not doc_markdown:
        # Pass explicit doc_id and rel_path to ocr_document for consistent logging
        doc_markdown = utils.ocr_document(
            client=client,
            path_obj=abs_source_path,
            logger=LOGGER,
            doc_id=doc_id,
            rel_path=rel_file_path,
        )
        if not doc_markdown:
            LOGGER.error(
                "❌ Analysis failed",
                extra={"doc_id": doc_id, "rel_path": rel_file_path},
            )
            return "❌ Failed to analyze document."

    # 2. Local Semantic Search (if not found in DB)
    if embedding_values is None:
        # Use RETRIEVAL_DOCUMENT to match the index schema (Symmetric Search)
        LOGGER.info("☁️ Generating query vector (Gemini-001)", extra={"doc_id": doc_id})
        query_res = utils.embed_text(
            client, doc_markdown, LOGGER, task_type="RETRIEVAL_DOCUMENT"
        )
        embedding_values = query_res.embeddings[0].values

    # 3. Iterative Retrieval for Valid Context
    # We want exactly N valid neighbors. Querying more if some are missing on disk.
    target_k = config.SETTINGS.ret_max_neighbors
    current_k = target_k
    max_k = 50  # Cap to prevent abuse

    valid_metadatas = []
    valid_documents = []

    while len(valid_metadatas) < target_k and current_k <= max_k:
        LOGGER.info(f"🔍 Querying {current_k} candidates...", extra={"doc_id": doc_id})

        results = vector_db.query(
            collection_name=config.SETTINGS.collection_name,
            query_embeddings=[embedding_values],
            n_results=current_k,
        )

        # Reset buckets to ensure order preservation from top rank
        valid_metadatas = []
        valid_documents = []

        # Process results
        if not results:
            break

        for res in results:
            md = res.metadata
            doc_content = res.content
            rel_path = md.get("rel_path")

            if not rel_path:
                continue

            # Check Blocklist
            if any(
                fnmatch.fnmatch(rel_path, pattern)
                for pattern in config.SETTINGS.retrieval_blocklist
            ):
                LOGGER.debug(f"🛑 Skipping blocklisted context: {rel_path}")
                continue

            # Check existence
            full_path = (Path(docs_base_dir) / rel_path).resolve()
            if not full_path.exists():
                LOGGER.debug(f"👻 Skipping missing file in context: {rel_path}")
                continue

            valid_metadatas.append(md)
            valid_documents.append(doc_content)
            if len(valid_metadatas) >= target_k:
                break

        if len(valid_metadatas) >= target_k:
            break

        # If we didn't find enough, verify if we even have enough in DB total
        # (This is hard to check efficiently without a count, but we just double k)
        if len(results) < current_k:
            # We retrieved everything available and it wasn't enough
            break

        current_k *= 2

    # 4. Decision reasoning via Gemini
    # Build a rich history context with filenames and content summaries
    history_items = []
    for i in range(len(valid_metadatas)):
        md = valid_metadatas[i]
        doc_content = valid_documents[i]
        # Truncate content if too long. Gemini Flash has a large context window,
        # so 10,000 chars is a good balance.
        limit = 10000
        snippet = (
            doc_content[:limit].replace("\n", " ") + "..."
            if len(doc_content) > limit
            else doc_content
        )

        history_items.append(
            {
                "filename": md.get("filename", "Unknown"),
                "relative_path": md.get("rel_path", "Unknown"),
                "content_summary": snippet,
            }
        )

    history = json.dumps(history_items, indent=2)
    history_ids = [item["relative_path"] for item in history_items]
    LOGGER.info(
        f"📜 History IDs used for context: {history_ids}", extra={"doc_id": doc_id}
    )

    response = utils.retry_with_backoff(LOGGER)(client.models.generate_content)(
        model=config.SETTINGS.analyzer_model_id,
        contents=[
            PROMPT_TEMPLATE.format(
                n_neighbors=config.SETTINGS.ret_max_neighbors, history=history
            ),
            doc_markdown,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=_FileMoveSuggestion
        ),
    )
    return response.parsed, abs_source_path, doc_markdown, embedding_values


def handle_move_action(res, abs_source_path, doc_markdown, embedding, base_dir, args):
    """
    Decides and executes the move action based on confidence and CLI arguments.
    """
    _cli_output("\n--- AI RECOMMENDATION ---")
    _cli_output(f"📂 FOLDER: {res.target_folder}")
    _cli_output(f"📄 NAME:   {res.new_filename}")
    _cli_output(f"💡 WHY:    {res.reasoning}")
    _cli_output(f"🎯 CONFIDENCE: {res.confidence_score * 100:.1f}%")

    # Generate dest path
    dest_dir = base_dir / res.target_folder
    dest_path = (dest_dir / res.new_filename).resolve()

    _cli_output(f"🚩 ABSOLUTE DEST: {dest_path}")

    # 4. Handle Move Execution
    moved_successfully = False

    if args.apply or args.auto_apply:
        # 4a. Auto-Apply Logic
        if args.auto_apply:
            if res.confidence_score > _AUTO_APPLY_CONFIDENCE_THRESHOLD:
                _cli_output(
                    f"⚡ [AUTO-APPLY] High confidence detected "
                    f"({res.confidence_score*100:.1f}%). Moving..."
                )
                moved_successfully = _perform_move(abs_source_path, dest_path)
            else:
                _cli_output(
                    f"⚠️  [AUTO-APPLY] Low confidence "
                    f"({res.confidence_score*100:.1f}%). "
                    f"Moving to quarantine '{config.SETTINGS.quarantine_folder}'."
                )
                quarantine_dest = (
                    base_dir / config.SETTINGS.quarantine_folder / res.new_filename
                )
                # We treat quarantine moves as 'successful' moves but they might be
                # blocked from indexing
                moved_successfully = _perform_move(abs_source_path, quarantine_dest)
                # Update dest_path for indexing check if needed, though likely blocked
                dest_path = quarantine_dest

        # 4b. Interactive Apply Logic
        elif args.apply:
            while True:
                try:
                    time.sleep(0.1)  # Ensure stderr is flushed/processed
                    _cli_output(
                        "\n⚠️  [y]es / [n]o / [e]dit suggested location? [y/N/e]"
                    )

                    # Force reading from /dev/tty
                    try:
                        with open("/dev/tty", "r") as tty:
                            confirm = tty.readline().strip().lower()
                    except OSError:
                        confirm = input(" > ").strip().lower()

                    if not confirm:
                        _cli_output(
                            "[DEBUG] Input was empty.", level="error", log_only=True
                        )

                    if confirm == "y":
                        moved_successfully = _perform_move(abs_source_path, dest_path)
                        break
                    elif confirm == "e":
                        _cli_output(
                            "✏️  Enter new relative path "
                            "(e.g. 'Financial/2024/Invoice.pdf'):"
                        )
                        try:
                            with open("/dev/tty", "r") as tty:
                                new_rel_input = tty.readline().strip()
                        except OSError:
                            new_rel_input = input(" > ").strip()

                        if not new_rel_input:
                            _cli_output("❌ Empty input. Try again.")
                            continue

                        # GUARDRAILS
                        try:
                            # 1. Reject Absolute Paths
                            input_path = Path(new_rel_input)
                            if input_path.is_absolute():
                                _cli_output(
                                    "❌ Absolute paths are not allowed. "
                                    "Please provide a path relative to "
                                    "the document library."
                                )
                                continue

                            # 2. Path Traversal Check
                            # Resolve the full path and ensure it's within base_dir
                            candidate_dest = (base_dir / input_path).resolve()
                            if not str(candidate_dest).startswith(
                                str(base_dir.resolve())
                            ):
                                _cli_output(
                                    "❌ Security Error: Path traversal detected. "
                                    "Target must be inside the document library."
                                )
                                continue

                            # Valid input
                            dest_path = candidate_dest
                            _cli_output(f"🔄 Destination updated to: {dest_path}")
                            # Loop continues to ask for confirmation on new path (y/n/e)

                        except Exception as e:
                            _cli_output(f"❌ Invalid path: {e}")
                            continue

                    else:
                        _cli_output("⏭️ Move skipped by user.")
                        break
                except Exception as e:
                    _cli_output(f"❌ Failed to interact/move file: {e}", level="error")
                    break
    else:
        _cli_output(f"🧪 [DRY RUN] Would move '{abs_source_path}' to '{dest_path}'")

    # 5. Index on Move
    if moved_successfully:
        try:
            # Calculate relative path for blocklist check
            try:
                rel_path = str(dest_path.relative_to(base_dir))
            except ValueError:
                rel_path = dest_path.name

            # Check Blocklist
            if any(
                fnmatch.fnmatch(rel_path, pattern)
                for pattern in config.SETTINGS.index_blocklist
            ):
                _cli_output(
                    f"🚫 Destination {rel_path} is in blocklist. Skipping indexing."
                )
                return

            _cli_output("🔄 Indexing moved file...")

            # Compute new hash
            file_hash = utils.generic.compute_file_hash(dest_path)
            doc_id = f"file_content_hash:{file_hash}"

            # Save to ChromaDB using SHARED logic
            vector_db = get_vector_db()

            utils.db.save_to_vector_store(
                vector_db,
                config.SETTINGS.collection_name,
                doc_id,
                dest_path,
                rel_path,
                doc_markdown,
                embedding,
                LOGGER,
            )

        except Exception as e:
            _cli_output(f"❌ Failed to index moved file: {e}", level="error")


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze PDF files and suggest filing locations."
    )
    parser.add_argument(
        "--file",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the PDF file(s) to analyze.",
    )
    parser.add_argument(
        "--docs-base-dir",
        type=str,
        default=str(config.SETTINGS.docs_base_dir),
        help=f"Base directory for documents (default: {config.SETTINGS.docs_base_dir})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Enable interactive move mode after analysis.",
    )
    parser.add_argument(
        "--auto-apply",
        action="store_true",
        default=False,
        help=(
            f"Automatically move files if confidence is > "
            f"{_AUTO_APPLY_CONFIDENCE_THRESHOLD*100:.0f}%%, otherwise skip."
        ),
    )
    return parser.parse_args()


def main():
    """Main entrypoint orchestrator."""
    args = parse_args()

    # Convert to Path object for clean manipulation
    base_dir = Path(args.docs_base_dir).expanduser().resolve()

    for target_arg in args.file:
        res_data = analyze_file(target_arg, base_dir)

        if isinstance(res_data, str):
            _cli_output(res_data, level="error")
        else:
            res, abs_source_path, doc_markdown, embedding = res_data
            handle_move_action(
                res, abs_source_path, doc_markdown, embedding, base_dir, args
            )

        _cli_output("-" * 40)


if __name__ == "__main__":
    main()
