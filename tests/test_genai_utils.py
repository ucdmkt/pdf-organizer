import unittest
from unittest.mock import MagicMock

from pdforganizer.utils import genai


class TestGenAIUtils(unittest.TestCase):

    def test_build_batch_ocr_request_success(self):
        """Test successful request build with a file URI (AI Studio or GCS)."""
        uploaded_file = MagicMock()
        # Mock the 'uri' attribute which is used for both AI Studio and GCS in the SDK
        uploaded_file.uri = "https://example.com/file"
        uploaded_file.mime_type = "application/pdf"

        jsonl, uri = genai.build_batch_ocr_request(uploaded_file, "doc1")

        self.assertEqual(uri, "https://example.com/file")
        self.assertEqual(jsonl["custom_id"], "doc1")
        self.assertEqual(
            jsonl["request"]["contents"][0]["parts"][0]["file_data"]["file_uri"],
            "https://example.com/file",
        )

    def test_build_batch_ocr_request_fail_no_uri(self):
        """Test failure when uploaded file has no URI."""
        uploaded_file = MagicMock()
        uploaded_file.uri = None

        jsonl, uri = genai.build_batch_ocr_request(uploaded_file, "doc1")

        self.assertIsNone(jsonl)
        self.assertIsNone(uri)


if __name__ == "__main__":
    unittest.main()
