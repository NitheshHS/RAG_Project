import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore


# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
    verify_ssl=False,
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter()
doc_splits=text_splitter.split_documents(docs)
print("doc splits:", doc_splits)
print(f"Split blog post into {len(doc_splits)} sub-documents.")

# create in-memory vector store and add documents
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = InMemoryVectorStore(embeddings)
document_ids=vector_store.add_documents(doc_splits)

print("vectorstore:", document_ids[:3])

# query the vector store
query = "What are agents in AI?"
retrieved_docs = vector_store.similarity_search(query, k=3)
print(f"Top 3 most similar documents to the query '{query}':")
for i, doc in enumerate(retrieved_docs):
    print(f"\nDocument {i + 1}:\n{doc.page_content}\n")
