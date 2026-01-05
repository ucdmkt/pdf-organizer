from pathlib import Path
from unittest.mock import patch

import pytest

from pdforganizer import indexer


class TestIndexer:
    @pytest.fixture
    def setup_indexer(self, mock_vector_db):
        self.mock_vector_db = mock_vector_db
        self.base_dir = Path("/tmp/docs")

    def test_is_blocklisted(self):
        """Test global blocklist check."""
        from pdforganizer import config

        new_settings = config.SETTINGS.model_copy(
            update={"index_blocklist": ["blocked/**"]}
        )
        with patch("pdforganizer.config.SETTINGS", new=new_settings):
            assert indexer._is_blocklisted("blocked/file.pdf")
            assert not indexer._is_blocklisted("safe/file.pdf")

    def test_scan_and_sync_idempotency(self, setup_indexer):
        """Test that moved files update metadata instead of duplication."""
        # Setup existing record with OLD path
        self.mock_vector_db.get.return_value = {
            "ids": ["file_content_hash:123"],
            "metadatas": [{"rel_path": "old/path.pdf", "filename": "path.pdf"}],
        }

        # We are scanning a file that hash 123, but is now at 'new/path.pdf'
        file_path = self.base_dir / "new/path.pdf"

        seen_ids = set()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.rglob", return_value=[file_path]),
            patch("pdforganizer.utils.generic.compute_file_hash", return_value="123"),
            patch("pathlib.Path.relative_to", return_value=Path("new/path.pdf")),
        ):

            # Generator should yield nothing because it found an update
            results = list(
                indexer._scan_and_sync_files(
                    self.base_dir, self.mock_vector_db, seen_ids
                )
            )

            assert len(results) == 0
            assert "file_content_hash:123" in seen_ids

            # Verify update was called
            self.mock_vector_db.upsert.assert_called()
            call_args = self.mock_vector_db.upsert.call_args
            assert call_args.kwargs["metadatas"][0]["rel_path"] == "new/path.pdf"
