#!/bin/bash
set -e

# Make sure we are in the project root
cd "$(dirname "$0")/.."

echo "🚀 Starting CI Checks..."
echo "📂 Current Directory: $(pwd)"
echo "👤 User: $(whoami)"
echo "🔧 uv version: $(uv --version)"
echo "🐍 python version: $(python --version)"


# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi


# 1. Setup Environment
echo "📦 Syncing environment..."
uv sync --all-extras --dev

# 2. Standard Linting (Black, Isort, Flake8)
echo "🔍 Checking imports and formatting..."
uv run isort . --check-only --profile black
uv run black . --check
uv run flake8 .

# 3. Type Checking
echo "🏷️ Checking types..."
uv run mypy pdforganizer

# 4. Unit Tests
echo "🧪 Running unit tests..."
uv run pytest
