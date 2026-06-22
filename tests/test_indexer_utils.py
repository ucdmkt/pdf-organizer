import hashlib
import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from pdforganizer import indexer

# Set up logging to avoid pollution
logging.basicConfig(level=logging.ERROR)

# Import modules to test
# We need to make sure we can import them.
# Assuming python path is set correctly or running from root works.


class TestIndexerUtils(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()
        self.mock_client = MagicMock()

    @patch("pdforganizer.indexer._load_batch_state")
    @patch("pdforganizer.config.SETTINGS")
    @patch("json.dump")
    @patch("builtins.open")
    def test_save_batch_state_merging(
        self, mock_open, mock_json_dump, mock_settings, mock_load
    ):
        """Test _save_batch_state with merge=True preserves existing data."""
        mock_settings.batch_state_file.exists.return_value = True
        mock_settings.model_dump.return_value = {}

        # Existing state
        mock_load.return_value = {"job_id": "ocr_1", "job_type": "OCR", "metadata": []}

        # Action: merge in texts
        indexer._save_batch_state({"texts": ["t1"]}, merge=True)

        # Verify
        args, _ = mock_json_dump.call_args
        saved_state = args[0]

        self.assertEqual(saved_state["job_id"], "ocr_1")
        self.assertEqual(saved_state["job_type"], "OCR")
        self.assertEqual(saved_state["texts"], ["t1"])

    def test_load_batch_state_config_invalidation(self):
        """Test _load_batch_state invalidates if config hash differs."""
        # 1. Setup Mock Config
        with (
            patch("pdforganizer.config.SETTINGS") as mock_settings,
            patch("pdforganizer.indexer._clear_batch_state") as mock_clear,
            patch("builtins.open", new_callable=MagicMock) as mock_open,
        ):

            mock_settings.batch_state_file.exists.return_value = True

            # Mock initial serialization (what was saved)
            # We simulate that the 'saved' hash was based on config_A
            # And the 'current' config is config_B

            config_A = {"ocr_model": "v1"}
            config_B = {"ocr_model": "v2"}  # Changed!

            # model_dump is called inside _compute_config_hash.
            # We need _load_batch_state -> _compute_config_hash
            # -> SETTINGS.model_dump() -> config_B
            mock_settings.model_dump.return_value = config_B

            # Pre-calculate hash of A for the saved file
            hash_A = hashlib.md5(
                json.dumps(config_A, sort_keys=True).encode("utf-8")
            ).hexdigest()

            saved_state = {"job_id": "old_job", "config_hash": hash_A, "metadata": []}
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(saved_state)
            )

            # 3. Call Load
            result = indexer._load_batch_state()

            # 4. Verify Invalidation
            self.assertIsNone(result)
            mock_clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
