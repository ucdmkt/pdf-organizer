from pathlib import Path
from unittest.mock import patch

import pytest

from pdforganizer import analyzer, config


class TestAnalyzer:
    @pytest.fixture
    def setup_mocks(self, mock_genai_client, mock_vector_db):
        self.mock_genai = mock_genai_client
        self.mock_vector_db = mock_vector_db
        self.base_dir = Path("/tmp/docs")
        self.input_file = self.base_dir / "input.pdf"

    def test_analyze_file_hash_reuse(self, setup_mocks):
        """Test that existing hash in DB reuses content and skips OCR."""
        # Setup existing record
        self.mock_vector_db.get.return_value = {
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
            self.mock_vector_db.get.assert_called()

            # Verify OCR was NOT called (upload_file should not be called)
            self.mock_genai.files.upload.assert_not_called()

    def test_analyze_file_blocklist_retrieval(self, setup_mocks):
        """Test that retrieval loop filters blocklisted files."""
        # Mock embeddings to force retrieval
        self.mock_vector_db.get.return_value = {"ids": []}

        # 1st Query returns blocked file
        # 2nd Query returns valid file
        def query_side_effect(collection_name, query_embeddings, n_results):
            if n_results <= config.SETTINGS.ret_max_neighbors:
                # Return SearchResult objects
                from pdforganizer.vectordb import SearchResult

                return [
                    SearchResult(
                        id="blocked",
                        content="B",
                        metadata={"rel_path": "unconfident/blocked.pdf"},
                        distance=0.1,
                    ),
                    SearchResult(
                        id="valid",
                        content="V",
                        metadata={"rel_path": "valid.pdf"},
                        distance=0.2,
                    ),
                ]
            return []

        self.mock_vector_db.query.side_effect = query_side_effect

        with (
            patch(
                "pdforganizer.utils.generic.compute_file_hash", return_value="newhash"
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.resolve", return_value=self.input_file),
            patch("pdforganizer.utils.ocr_document", return_value="OCR Content"),
        ):

            # Override blocklist
            new_settings = config.SETTINGS.model_copy(
                update={"retrieval_blocklist": ["unconfident/**"]}
            )
            with patch("pdforganizer.config.SETTINGS", new=new_settings):
                analyzer.analyze_file("input.pdf", str(self.base_dir))

                # Check that generate_content was called with history
                # that excludes the blocked file
                call_args = self.mock_genai.models.generate_content.call_args
                prompt = call_args[1]["contents"][0]  # or call_args.kwargs

                # We expect valid.pdf in prompt, but NOT unconfident/blocked.pdf
                assert "valid.pdf" in prompt
                assert "unconfident/blocked.pdf" not in prompt

    def test_handle_move_action_interactive_edit_traversal(self):
        """Test that interactive edit rejects path traversal."""
        base_dir = Path("/tmp/docs").resolve()
        source = base_dir / "input.pdf"
        # dest = base_dir / "target.pdf"  # Unused

        # Mock res object
        class MockRes:
            target_folder = "."
            new_filename = "target.pdf"
            reasoning = "Test"
            confidence_score = 0.95

        # Mock args
        class MockArgs:
            apply = True
            auto_apply = False

        # Input simulation:
        # 1. 'e' (edit)
        # 2. '../../etc/passwd' (traversal attempt)
        # 3. 'n' (abort loop)
        inputs = ["e", "../../etc/passwd", "n"]

        with (
            patch("builtins.input", side_effect=inputs),
            patch("builtins.open", side_effect=OSError),  # Force input() usage
            patch("pdforganizer.analyzer._perform_move") as mock_move,
            # We mock _cli_output to verify error messages
            patch("pdforganizer.analyzer._cli_output") as mock_io,
        ):
            analyzer.handle_move_action(
                MockRes(), source, "md", [], base_dir, MockArgs()
            )

            # mock_move should NOT be called
            mock_move.assert_not_called()

            # Verify security waring was logged
            calls = [str(c) for c in mock_io.mock_calls]
            assert any("Security Error" in c for c in calls)

    def test_handle_move_action_interactive_edit_whitespace(self):
        """Test that interactive edit handles filenames with spaces."""
        base_dir = Path("/tmp/docs").resolve()
        source = base_dir / "input.pdf"

        # Mock res object
        class MockRes:
            target_folder = "."
            new_filename = "target.pdf"
            reasoning = "Test"
            confidence_score = 0.95

        # Mock args
        class MockArgs:
            apply = True
            auto_apply = False

        # Input simulation:
        # 1. 'e' (edit)
        # 2. 'New Folder/My File.pdf' (input with spaces)
        # 3. 'y' (confirm)
        inputs = ["e", "New Folder/My File.pdf", "y"]

        with (
            patch("builtins.input", side_effect=inputs),
            patch("builtins.open", side_effect=OSError),
            patch("pdforganizer.analyzer._perform_move") as mock_move,
            patch("pdforganizer.analyzer._cli_output"),
        ):
            analyzer.handle_move_action(
                MockRes(), source, "md", [], base_dir, MockArgs()
            )

            # Expect move to resolved path with spaces
            expected_dest = base_dir / "New Folder/My File.pdf"
            mock_move.assert_called_with(source, expected_dest)
