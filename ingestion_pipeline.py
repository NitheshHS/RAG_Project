# load documents
import os.path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def load_documents(source_path: str):
    if not os.path.exists(source_path):
        raise FileExistsError(f"{source_path} not exists")
    loader = DirectoryLoader(path=source_path,
                             glob="*.txt",
                             loader_cls=TextLoader, )
    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"{source_path} not exists")
    for i, doc in enumerate(documents[:5]):
        print(f"source: {doc.metadata["source"]}")
        print(f"Document {i + 1}:\n{doc.page_content}\n")
    return documents


# split documents
def split_documents(documents):
    splitter = CharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    # split documents
    chunks = splitter.split_documents(documents)
    print(f"chunks: {len(chunks)}")
    for chunk in chunks[:5]:
        print(f"Sample chunk:\n{chunk.page_content}\n")
    return chunks


#  create vector store
def create_vector_store(chunks, folder_path="db/faiss_index"):
    print("creating vector store")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path=folder_path)
    print("vectorstore saved to:", folder_path)
    return vector_store


if __name__ == "__main__":
    document = load_documents(source_path="documents")
    chunks = split_documents(document)
    create_vector_store(chunks)
