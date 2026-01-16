# RAG PDF CLI

A command-line tool that uses Retrieval-Augmented Generation (RAG) to answer questions about PDF documents using Google's Gemini AI.

## Features

- Load and parse PDF documents
- Split documents into chunks for efficient retrieval
- Create vector embeddings using Google Generative AI
- Store embeddings in a FAISS vector store
- Interactive Q&A interface powered by Gemini

## Requirements

- Python 3.10+
- Google Gemini API key

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RagProject
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Usage

Run the CLI with a PDF file:

```bash
python rag_pdf_cli.py /path/to/your/document.pdf
```

Once the index is built, you can ask questions interactively:

```
Building index (first time may take a bit)...
Ready. Ask questions. Type 'exit' to quit.

Q: What is this document about?

A: This document is about...
Sources pages: [1, 2, 3]

Q: exit
```

## How It Works

1. **PDF Loading**: The PDF is loaded using `PyPDFLoader` from LangChain
2. **Text Splitting**: Documents are split into chunks (900 chars with 150 overlap) using `RecursiveCharacterTextSplitter`
3. **Embedding**: Chunks are embedded using Google's `embedding-001` model
4. **Vector Store**: Embeddings are stored in a FAISS index for fast similarity search
5. **Retrieval**: When you ask a question, the top 4 most relevant chunks are retrieved
6. **Generation**: Gemini generates an answer based only on the retrieved context

## Configuration

You can modify these parameters in the code:

- `chunk_size`: Size of text chunks (default: 900)
- `chunk_overlap`: Overlap between chunks (default: 150)
- `k`: Number of chunks to retrieve (default: 4)
- `model`: LLM model for generation (default: "gemini-3-pro-preview")

## Dependencies

- `langchain-community` - Document loaders and vector stores
- `langchain-google-genai` - Google Gemini integration
- `langchain-text-splitters` - Text splitting utilities
- `pypdf` - PDF parsing
- `faiss-cpu` - Vector similarity search
- `pydantic` - Data validation

## License

MIT License

