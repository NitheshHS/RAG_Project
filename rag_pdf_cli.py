import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say: "I couldn't find that in the PDF."

Context:
{context}

Question: {question}
Answer:
""")

import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("GEMINI API KEY")


def build_index(pdf_path: str, save_path: str = "db/faiss_index"):
    # index the PDF into chunks and create a FAISS vector store
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # one Document per page
    # split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    # split documents
    chunks = splitter.split_documents(docs)
    # create FAISS vector store
    # embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', api_key=SecretStr(os.environ['GOOGLE_API_KEY']))
    db = FAISS.from_documents(chunks, embeddings)

    # Save the index to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    db.save_local(save_path)
    print(f"Index saved to {save_path}")

    return db

def answer_question(db, question: str, k: int = 4):
    # Retrieve top-k chunks
    retrieved = db.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in retrieved)

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview",
                                 temperature=1.0,  # Gemini 3.0+ defaults to 1.0
                                 max_tokens=None,
                                 timeout=None,
                                 max_retries=2, )
    msg = PROMPT.format_messages(context=context, question=question)
    resp = llm.invoke(msg)

    return resp.content, retrieved


def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_pdf_cli.py /path/my_pdf.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print("Building index (first time may take a bit)...")
    db = build_index(pdf_path)
    print("Ready. Ask questions. Type 'exit' to quit.\n")

    while True:
        q = input("Q: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        ans, sources = answer_question(db, q, k=4)
        print("\nA:", ans)

        # Show which pages were used (helps trust/debug)
        pages = sorted(set(s.metadata.get("page") for s in sources if "page" in s.metadata))
        if pages:
            print("Sources pages:", pages)
        print()


if __name__ == "__main__":
    main()
