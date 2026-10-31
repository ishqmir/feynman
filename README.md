# Feynman

A simple RAG app built with Streamlit, LangChain, ChromaDB, and Ollama to ask questions about your research papers and textbooks locally.

## Prerequisites

```bash
ollama pull qwen2.5:1.5b
ollama pull mxbai-embed-large
```

## How to Run

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the Streamlit application:
   ```bash
   uv run streamlit run app.py
   ```
