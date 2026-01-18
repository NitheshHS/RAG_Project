# RAG Project

A comprehensive Retrieval-Augmented Generation (RAG) system with multiple pipelines for document ingestion, retrieval, and conversational Q&A.

## Features

- **Document Ingestion** - Load and process text files or PDFs into vector embeddings
- **FAISS Vector Store** - Efficient similarity search for document retrieval
- **History-Aware Chat** - Maintains conversation context for follow-up questions
- **PDF Q&A CLI** - Interactive command-line tool for querying PDF documents
- **Web Scraping** - Index content from web pages
- **Multiple LLM Support** - Works with OpenAI GPT-4o-mini and Google Gemini

## Project Structure

```
RagProject/
├── ingestion_pipeline.py       # Ingest text documents into FAISS
├── retrieval_pipeline.py       # Simple RAG retrieval and response
├── history_aware_generation.py # Conversational RAG with chat history
├── rag_pdf_cli.py              # PDF Q&A command-line tool (Google Gemini)
├── indexing/
│   └── rag_indexing.py         # Web scraping and in-memory indexing
├── documents/
│   └── sample_documents.txt    # Sample documents for testing
├── db/
│   └── faiss_index/            # Persisted FAISS vector store
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- OpenAI API key (for OpenAI-based pipelines)
- Google API key (for Gemini-based PDF CLI)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RagProject
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   ```

## Usage

### 1. Ingest Documents

Load text files from the `documents/` folder and create a FAISS index:

```bash
python ingestion_pipeline.py
```

This will:
- Load all `.txt` files from `documents/`
- Split them into chunks (900 chars, 150 overlap)
- Create embeddings using OpenAI `text-embedding-3-large`
- Save the FAISS index to `db/faiss_index/`

### 2. Simple Retrieval

Query the indexed documents:

```bash
python retrieval_pipeline.py
```

```
Enter a query: What is RAG?
---Generated Response---
RAG (Retrieval-Augmented Generation) is a technique that...
```

### 3. History-Aware Chat

Start an interactive chat with conversation memory:

```bash
python history_aware_generation.py
```

```
Welcome to the History-Aware RAG Chat!
Type 'exit' to end the chat.
You: What is RAG?
Answer: RAG is a technique that enables LLMs to...

You: How does it reduce hallucinations?
Answer: Based on our previous discussion about RAG...

You: exit
Ending the chat. Goodbye!
```

### 4. PDF Q&A CLI (Google Gemini)

Query any PDF document interactively:

```bash
python rag_pdf_cli.py /path/to/your/document.pdf
```

```
Building index (first time may take a bit)...
Index saved to db/faiss_index
Ready. Ask questions. Type 'exit' to quit.

Q: What is this document about?
A: This document discusses...
Sources pages: [1, 2, 3]

Q: exit
```

### 5. Web Scraping & Indexing

Index content from web pages:

```bash
python indexing/rag_indexing.py
```

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Documents  │────▶│  Chunking   │────▶│  Embeddings │
│  (txt/pdf)  │     │  (900 char) │     │  (OpenAI)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│     LLM     │◀────│   FAISS     │
│  Generation │     │ (GPT-4o)    │     │  Retrieval  │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Ingestion**: Documents are loaded, split into chunks, and embedded
2. **Storage**: Embeddings are stored in FAISS vector database
3. **Retrieval**: User query is embedded and top-k similar chunks are retrieved
4. **Generation**: LLM generates an answer using retrieved context

## Configuration

| Component | Model | Description |
|-----------|-------|-------------|
| Embeddings (OpenAI) | `text-embedding-3-large` | High-quality embeddings |
| Embeddings (Google) | `models/embedding-001` | Google embedding model |
| Chat (OpenAI) | `gpt-4o-mini` | Fast, cost-effective responses |
| Chat (Google) | `gemini-3-pro-preview` | Google's latest model |
| Chunk Size | 900 characters | Document splitting |
| Chunk Overlap | 150 characters | Context continuity |
| Top-K Retrieval | 4 documents | Number of relevant docs |

## Dependencies

```
langchain
langchain-openai
langchain-google-genai
langchain-community
langchain-core
langchain-text-splitters
faiss-cpu
pypdf
beautifulsoup4
python-dotenv
```

## Troubleshooting

### SSL Certificate Error
If you encounter SSL errors, try:
```bash
/Applications/Python\ 3.14/Install\ Certificates.command
```
Or disconnect from VPN/corporate network.

### Missing FAISS Index
Run the ingestion pipeline first:
```bash
python ingestion_pipeline.py
```

### API Key Not Set
```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

