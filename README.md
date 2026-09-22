# RAG Pipeline: From Data Ingestion to Vector DB

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built from scratch — covering document ingestion, chunking, embedding, vector storage, retrieval, LLM-based answer generation, agentic workflows, and evaluation.

## Overview

This repo walks through building a RAG system step by step, moving from raw document loading all the way to an agentic, evaluated pipeline. Each notebook focuses on one stage of the pipeline, so you can follow the progression from basic ingestion to a production-style setup.

## Pipeline Stages

**1. Data Ingestion** (`notebook/document.ipynb`)
- Working with LangChain's `Document` structure and metadata
- Loading text files with `TextLoader` and `DirectoryLoader`
- Loading PDFs with `PyPDFLoader` and `PyMuPDFLoader`

**2. Ingestion to Vector DB** (`notebook/pdf_loader (1).ipynb`)
- Batch PDF processing and recursive character text splitting (chunking)
- Generating embeddings with `sentence-transformers` (`all-MiniLM` family)
- Storing embeddings in ChromaDB via a custom `VectorStore` class
- A `RAGRetriever` class for similarity search over the vector store
- Answer generation with Groq-hosted LLMs (`gemma2-9b-it`) via `langchain-groq`
- An enhanced pipeline with streaming, citations, chat history, and summarization

**3. Alternative Vector Store — Typesense** (`src/typesense.ipynb`)
- Using Typesense as both a search engine and vector store
- Importing structured data (`books.jsonl`) and running keyword/faceted search
- A LangChain + Typesense + Groq RAG flow as an alternative to ChromaDB/FAISS

**4. Agentic RAG** (`agenticrag/1-agenticrag.ipynb`)
- A LangGraph state machine (`AgentState`) that decides *whether* retrieval is needed before answering
- Nodes for retrieval decision, document retrieval, and answer generation
- Conditional graph edges based on the retrieval decision
- Built on `ChatOpenAI` (GPT-4.1) and `OpenAIEmbeddings`

**5. Evaluation** (`src/1-rag_evaluation.ipynb`)
- Chatbot and RAG evaluation using an LLM-as-judge approach
- Metrics: **Correctness** (vs. ground truth), **Relevance** (response vs. input), **Groundedness** (response vs. retrieved context), and **Retrieval Relevance** (retrieved docs vs. input)

**6. Example Application** (`src/app.py`)
- A minimal script tying the pipeline together: document loading → FAISS vector store → `RAGSearch.search_and_summarize()`

## Tech Stack

- **Orchestration:** LangChain, LangGraph
- **Vector Stores:** FAISS, ChromaDB, Typesense
- **Embeddings:** Sentence-Transformers
- **LLMs:** Groq (Gemma2-9B-IT), OpenAI (GPT-4.1)
- **Document Loading:** PyPDF, PyMuPDF

## Project Structure

```
.
├── notebook/
│   ├── document.ipynb              # Stage 1: Data ingestion basics
│   ├── pdf_loader (1).ipynb        # Stage 2: Chunking, embedding, ChromaDB, retrieval + generation
│   └── 1-langchain-document-components.svg
├── src/
│   ├── app.py                      # Example end-to-end script (FAISS-based)
│   ├── typesense.ipynb             # Stage 3: Typesense-based RAG
│   ├── 1-rag_evaluation.ipynb      # Stage 5: RAG evaluation metrics
│   ├── books.jsonl                 # Sample dataset for Typesense demo
│   └── requirements.txt
└── agenticrag/
    └── 1-agenticrag.ipynb          # Stage 4: LangGraph agentic RAG
```

## Getting Started


1. **Install dependencies**
   ```bash
   pip install -r src/requirements.txt
   ```

2. **Set up environment variables**
   Create a `.env` file with the API keys needed for the notebooks you plan to run:
   ```
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Run the notebooks in order** to follow the pipeline from ingestion through evaluation, or jump directly to the stage you're interested in.

## Notes

- This is a learning/portfolio project exploring different RAG architectures and vector stores side by side (FAISS, ChromaDB, Typesense) rather than a single production pipeline.
- `src/app.py` references `src/data_loader.py`, `src/vectorstore.py`, and `src/search.py` modules — these are still being modularized out of the notebooks and are a work in progress.

## Author

**Avan Kumar** — [GitHub](https://github.com/avankumar23)
