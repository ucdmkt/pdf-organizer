import os
from pathlib import Path

import pytest
import yaml

from pdforganizer import config


class TestConfig:
    @pytest.fixture
    def clean_env(self):
        """Reset config singleton and env vars."""
        # Reset the singleton if implemented that way, or reload module.
        # For this design, we assume a load_config function returns a new object.
        original_env = os.environ.get("GOOGLE_API_KEY")
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        yield

        if original_env:
            os.environ["GOOGLE_API_KEY"] = original_env

        if "GOOGLE_CLOUD_PROJECT" in os.environ:
            del os.environ["GOOGLE_CLOUD_PROJECT"]

    def test_default_config(self, clean_env):
        """Test default values when no config file exists."""
        # Ensure no config file is picked up by pointing to non-existent
        cfg = config.load_config(Path("/non/existent/config.yaml"))

        assert cfg.ret_max_neighbors == 7
        assert "unconfident/**" in cfg.retrieval_blocklist
        assert cfg.analyzer_model_id == "models/gemini-3-flash-preview"
        assert cfg.ocr_model_id == "models/gemini-2.5-flash-lite"
        # Confirm no hardcoded fallback
        assert cfg.google_api_key is None
        assert cfg.google_cloud_project is None
        assert cfg.google_cloud_location == "us-central1"

    def test_yaml_loading(self, clean_env, tmp_path):
        """Test loading from YAML file."""
        config_data = {
            "ret_max_neighbors": 10,
            "analyzer_model_id": "gemini-1.5-pro",
            "index_blocklist": ["foo/**"],
            "docs_base_dir": "~/docs_expansion_test",
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = config.load_config(config_file)

        assert cfg.ret_max_neighbors == 10
        assert cfg.analyzer_model_id == "gemini-1.5-pro"
        assert "foo/**" in cfg.index_blocklist
        assert cfg.docs_base_dir == Path("~/docs_expansion_test").expanduser().resolve()
        assert "foo/**" in cfg.index_blocklist
        # Check derived values: index_blocklist MUST be in retrieval_blocklist,
        assert "foo/**" in cfg.retrieval_blocklist  # index blocklist included

    def test_env_var_secret(self, clean_env):
        """Test API key loading from environment."""
        os.environ["GOOGLE_API_KEY"] = "TEST_KEY"
        cfg = config.load_config(Path("/non/existent"))
        cfg = config.load_config(Path("/non/existent"))
        assert cfg.google_api_key == "TEST_KEY"

    def test_vertex_config_loading(self, clean_env):
        """Test GCP Project loading from environment."""
        os.environ["GOOGLE_CLOUD_PROJECT"] = "my-gcp-project"
        cfg = config.load_config(Path("/non/existent"))
        assert cfg.google_cloud_project == "my-gcp-project"
        assert cfg.google_cloud_location == "us-central1"

    def test_cwd_loading(self, clean_env, tmp_path, monkeypatch):
        """Test loading from ./config.yaml."""
        # Mock CWD to be tmp_path
        monkeypatch.chdir(tmp_path)

        config_data = {"ret_max_neighbors": 55}
        with open(tmp_path / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

        cfg = config.load_config()
        assert cfg.ret_max_neighbors == 55

    def test_db_path_resolution(self, clean_env, tmp_path):
        """Test db_path resolution relative to config file."""
        # Case 1: Relative path in config
        config_data = {"db_path": "my_db"}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = config.load_config(config_file)
        # Should be resolved relative to config_file's parent (tmp_path)
        assert cfg.db_path == tmp_path / "my_db"

        # Case 2: Absolute path in config
        abs_path = tmp_path / "abs_db"
        config_data = {"db_path": str(abs_path)}  # Str required for yaml dump of Path
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = config.load_config(config_file)
        assert cfg.db_path == abs_path
