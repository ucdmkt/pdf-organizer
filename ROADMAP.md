# Roadmap

## ⚠️ Known Constraints

- **Vertex AI Batch Embedding Support**: The `gemini-embedding-001` model is **not supported** by the Vertex AI Batch API (returns `404 NOT_FOUND`). See [official documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings#batch_predictions).
  - **Constraint**: The project requires `gemini-embedding-001` for embedding (due to vector compatibility or hard requirement).
  - **Blocker**: This creates a hard blocker for running Batch Embeddings on Vertex AI with this model.
  - **Workarounds Explored**:
    - Switching model to `text-embedding-004` (works but rejected by user requirements).
    - Online Fallback (works functionally but rejected due to architectural preferences/revert request).
  - **Resolution**: Vertex AI Batch mode is currently incompatible with `gemini-embedding-001`. Use AI Studio or a supported model like `text-embedding-004` if requirements change.

## 🚀 Priority 1: Remote ChromaDB Support

- [ ] **Remote Client Implementation**
  - Implement `RemoteChromaClient` in `pdforganizer/vectordb`.
  - Update `pdforganizer/vectordb/__init__.py` to select client based on config.
- [ ] **Configuration Updates**
  - Add `vectordb_provider` (local vs remote).
  - Add connection settings: `chroma_server_host`, `chroma_server_port`, `chroma_auth_token`.
- [ ] **Testing**
  - Add unit tests for `RemoteChromaClient`.

## 🛠️ Priority 2: Multi-Provider Validation

- [ ] **New Vector Providers**
  - Add support for **Qdrant** or **Milvus** to validate the `VectorDBClient` abstraction layer.

## 🤖 Priority 3: Local LLM / AI Abstraction

- [ ] **AI Provider Interface**
  - Abstract `utils.genai` into a provider-agnostic interface (similar to `VectorDBClient`).
  - Decouple from explicit `google.genai` dependencies.
- [ ] **Ollama Support**
  - Implement support for local models via **Ollama** (e.g., Llama 3, Mistral) for Analyzer logic.
  - Support local embeddings (e.g., `nomic-embed-text`).
