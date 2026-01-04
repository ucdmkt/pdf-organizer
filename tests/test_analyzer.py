from pathlib import Path
from unittest.mock import patch

import pytest

from pdforganizer import analyzer, config


class TestAnalyzer:
    @pytest.fixture
    def setup_mocks(self, mock_genai_client, mock_chroma_client):
        self.mock_genai = mock_genai_client
        self.mock_chroma, self.mock_collection = mock_chroma_client
        self.base_dir = Path("/tmp/docs")
        self.input_file = self.base_dir / "input.pdf"

    def test_analyze_file_hash_reuse(self, setup_mocks):
        """Test that existing hash in DB reuses content and skips OCR."""
        # Setup existing record
        self.mock_collection.get.return_value = {
            "ids": ["file_content_hash:123"],
            "documents": ["Cached Content"],
            "embeddings": [[0.1] * 768],
            "metadatas": [{"filename": "input.pdf", "rel_path": "input.pdf"}],
        }

        with (
            patch("pdforganizer.utils.generic.compute_file_hash", return_value="123"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.resolve", return_value=self.input_file),
        ):

            # Run
            analyzer.analyze_file("input.pdf", str(self.base_dir))

            # Verify Collection Get was called
            self.mock_collection.get.assert_called()

            # Verify OCR was NOT called (upload_file should not be called)
            self.mock_genai.files.upload.assert_not_called()

    def test_analyze_file_blocklist_retrieval(self, setup_mocks):
        """Test that retrieval loop filters blocklisted files."""
        # Mock embeddings to force retrieval
        self.mock_collection.get.return_value = {"ids": []}

        # 1st Query returns blocked file
        # 2nd Query returns valid file
        def query_side_effect(query_embeddings, n_results, include):
            if n_results <= config.settings.ret_max_neighbors:
                return {
                    "ids": [["blocked", "valid"]],
                    "metadatas": [
                        [
                            {"rel_path": "unconfident/blocked.pdf"},
                            {"rel_path": "valid.pdf"},
                        ]
                    ],
                    "documents": [["B", "V"]],
                }
            return {"ids": []}

        self.mock_collection.query.side_effect = query_side_effect

        with (
            patch(
                "pdforganizer.utils.generic.compute_file_hash", return_value="newhash"
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.resolve", return_value=self.input_file),
            patch("pdforganizer.utils.ocr_document", return_value="OCR Content"),
        ):

            # Override blocklist
            with patch(
                "pdforganizer.config.settings.retrieval_blocklist", ["unconfident/**"]
            ):
                analyzer.analyze_file("input.pdf", str(self.base_dir))

                # Check that generate_content was called with history
                # that excludes the blocked file
                call_args = self.mock_genai.models.generate_content.call_args
                prompt = call_args[1]["contents"][0]  # or call_args.kwargs

                # We expect valid.pdf in prompt, but NOT unconfident/blocked.pdf
                assert "valid.pdf" in prompt
                assert "unconfident/blocked.pdf" not in prompt
