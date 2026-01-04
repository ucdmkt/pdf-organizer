import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from pdforganizer.utils import generic, resilience


class TestUtils:
    def test_compute_file_hash(self):
        """Test SHA256 file hashing."""
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(b"content")
            tmp.flush()
            hash1 = generic.compute_file_hash(Path(tmp.name))

            tmp.seek(0)
            tmp.write(b"content")
            tmp.flush()
            hash2 = generic.compute_file_hash(Path(tmp.name))

            assert hash1 == hash2
            # sha256 of "content"
            expected = (
                "ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73"
            )
            assert hash1 == expected

    def test_should_retry_on_exception(self):
        """Test retry predicate logic."""
        # 429
        e1 = MagicMock()
        e1.status_code = 429
        assert resilience.should_retry_on_exception(e1)

        # Network timeout (simulated by checking class check logic logic)
        # Note: mocking httpx exception exact types is tricky without unnecessary
        # deps imports in test, so we rely on string check for simplicity or
        # generic mock behavior if possible.
        # But we can verify the text based check easily.

        e2 = Exception("Connection timed out")
        assert resilience.should_retry_on_exception(e2)

        e3 = Exception("Resource Exhausted")
        assert resilience.should_retry_on_exception(e3)

        e4 = Exception("ValueError")
        assert not resilience.should_retry_on_exception(e4)
