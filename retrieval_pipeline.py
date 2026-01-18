from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os


def retrieve_documents(query: str):
    """Retrieve relevant documents and generate a response using RAG."""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    print("embeddings:", embeddings)

    # Check if FAISS index exists
    faiss_path = "db/faiss_index"
    if not os.path.exists(faiss_path):
        print(f"Error: FAISS index not found at '{faiss_path}'. Please create the index first.")
        return

    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    print("vectorstore:", db)

    retriever = db.as_retriever(search_kwargs={"k": 4})
    relevant_documents = retriever.invoke(query)

    if not relevant_documents:
        print("No relevant documents found.")
        return

    for doc in relevant_documents:
        print(f"Document:\n{doc.page_content}\n")
        print("content metadata:", doc.metadata)
        print("document id:", doc.id)
        print("document source:", doc.metadata.get("source"))

    # Join documents with clear separators
    documents_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_documents])

    combined_input = f"""Based on the following documents, please answer the question: {query}

Documents:
{documents_text}

Please provide a clear, helpful answer using the information above."""

    print("combined input:", combined_input)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content="You are a helpful assistant that helps people find information."),
        HumanMessage(content=combined_input)
    ]
    response = model.invoke(messages)

    print("---Generated Response---")
    print(response.content)
    return response.content


if __name__ == "__main__":
    query = input("Enter a query: ")
    retrieve_documents(query)
