# Roadmap

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
