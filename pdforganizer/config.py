import logging
import os

import yaml
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path  # noqa: E402
from typing import List, Optional  # noqa: E402

from pydantic import BaseModel, Field, field_validator  # noqa: E402

LOGGER = logging.getLogger(__name__)

# Default Constants
DEFAULT_INDEX_BLOCKLIST: List[str] = []


class AppConfig(BaseModel):
    """Application Configuration Schema"""

    # --- API CONFIG ---
    google_api_key: Optional[str] = Field(
        default=None, description="Google API Key. Defaults to env var GOOGLE_API_KEY."
    )

    # --- MODEL CONFIG ---
    ocr_model_id: str = "models/gemini-2.5-flash-lite"
    analyzer_model_id: str = "models/gemini-3-flash-preview"
    embed_model_id: str = "models/gemini-embedding-001"
    embed_dimension: int = 768

    # --- DB CONFIG ---
    db_path: Path = Field(default_factory=lambda: Path.cwd() / "hybrid_semantic_db")
    collection_name: str = "native_vault"
    docs_base_dir: Path = Field(
        default_factory=lambda: Path("~/documents/").expanduser().resolve()
    )

    # --- BLOCKLISTS ---
    index_blocklist: List[str] = Field(
        default=[],
        description=(
            "Glob patterns to exclude from indexing (e.g. ['**/private/*', '*.tmp'])"
        ),
    )
    retrieval_blocklist: List[str] = Field(
        default=[],
        description=(
            "Glob patterns to exclude from being used as context "
            "(automatically includes index_blocklist)"
        ),
    )

    # --- SETTINGS ---
    ret_max_neighbors: int = 7
    quarantine_folder: str = "unconfident"
    batch_state_file: Path = Field(
        default_factory=lambda: Path("~/.local/state/pdforganizer/batch_state.json")
        .expanduser()
        .resolve()
    )

    def model_post_init(self, __context):
        """Post-initialization to set derived values and Env vars logic."""
        # 1. Load API Key from Env if not provided
        if not self.google_api_key:
            self.google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not self.google_api_key:
                # API Key is mandatory for GenAI features, but we don't crash
                # immediately to allow checking help or partial functionality.
                LOGGER.warning(
                    "⚠️ GOOGLE_API_KEY not found in config or environment. "
                    "GenAI features will fail."
                )

        # 2. Derive Retrieval Blocklist
        # Logic: base (user provided or default) + index_blocklist.
        # This ensures users don't have to duplicate rules.
        user_retrieval_rules = (
            self.retrieval_blocklist
            if self.retrieval_blocklist is not None
            else ["unconfident/**"]
        )

        # Merge ensuring uniqueness, preserving order (index rules first)
        merged = list(self.index_blocklist)
        for rule in user_retrieval_rules:
            if rule not in merged:
                merged.append(rule)

        self.retrieval_blocklist = merged

    @field_validator("batch_state_file", "db_path", "docs_base_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(v).expanduser().resolve()


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Loads configuration from YAML file.
    Search Order:
    1. Explicit path
    2. ./config.yaml
    3. ~/.config/pdforganizer/config.yaml
    4. Defaults
    """
    selected_path = None  # 1. Explicit path arg

    if config_path and config_path.exists():
        selected_path = config_path
    elif (Path.cwd() / "config.yaml").exists():
        selected_path = Path.cwd() / "config.yaml"
    elif (Path.home() / ".config/pdforganizer/config.yaml").exists():
        selected_path = Path.home() / ".config/pdforganizer/config.yaml"

    if selected_path:
        try:
            with open(selected_path, "r") as f:
                data = yaml.safe_load(f) or {}

                # Resolve relative db_path against config file directory
                if "db_path" in data and data["db_path"]:
                    p = Path(data["db_path"]).expanduser()
                    if not p.is_absolute():
                        # Treat as relative to the configuration file directory
                        data["db_path"] = (selected_path.parent / p).resolve()

                LOGGER.info(f"🔧 Loading config from {selected_path}")
                return AppConfig(**data)
        except Exception as e:
            LOGGER.warning(
                f"⚠️ Failed to load config from {selected_path}: {e}. Using defaults."
            )

    return AppConfig()


# Global Settings Instance
# Users can interact with this, or reload it if they want dynamic reloading
settings = load_config()
