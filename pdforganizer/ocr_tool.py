"""Standalone CLI tool for performing OCR on individual PDF documents."""

import argparse
import sys
from pathlib import Path

from pdforganizer import utils

LOGGER = utils.setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform OCR on a single PDF document."
    )
    parser.add_argument("input_file", type=str, help="Path to the input PDF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output Markdown file. If not provided, prints to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        LOGGER.error(f"Input file not found: {input_path}")
        sys.exit(1)

    LOGGER.info(f"Processing file: {input_path}")

    # Initialize Client
    try:
        client = utils.get_genai_client()
    except Exception as e:
        LOGGER.error(f"Failed to initialize GenAI client: {e}")
        sys.exit(1)

    # Perform OCR
    markdown_content = utils.ocr_document(client, input_path, LOGGER)

    if not markdown_content:
        LOGGER.error("OCR failed to extract content.")
        sys.exit(1)

    # Handle Output
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            LOGGER.info(f"✅ OCR result saved to: {output_path}")
        except Exception as e:
            LOGGER.error(f"Failed to write output file: {e}")
            sys.exit(1)
    else:
        # Print to stdout
        print(markdown_content)


if __name__ == "__main__":
    main()
