from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

chat_history = []
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

def ask_question(user_question: str):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # check if there is chat history
    if chat_history:
        # build the message with history
        message = [
            SystemMessage(content="Given the conversation history, "
                                  "reformulate the user's question into a standalone search query. Only return the query, nothing else."),
            *chat_history,
            HumanMessage(content=f"New User question: {user_question}")
        ]
        # invoke the model with history
        result = model.invoke(message)
        # get the search question from the response
        search_question = result.content
    else:
        # if its the first question, use it directly
        search_question = user_question

    # load existing FAISS index
    db = FAISS.load_local(folder_path="db/faiss_index", embeddings=embedding,
                          allow_dangerous_deserialization=True)
    print("vectorstore:", db)
    # retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 4})
    # invoke retriever to get relevant documents
    relevant_documents = retriever.invoke(search_question)

    if not relevant_documents:
        print("No relevant documents found.")
        return

    # print the retrieved documents
    for doc in relevant_documents:
        print(f"Document:\n{doc.page_content}\n")
        print("content metadata:", doc.metadata)
        print("document id:", doc.id)
        print("document source:", doc.metadata.get("source"))

    # build context for final answer
    combined_input = f"""Based on the following documents, please answer the question: {user_question},
    Documents: {"\n\n---\n\n".join([doc.page_content for doc in relevant_documents])}
    Please provide a clear, helpful answer using the information above.
    """
    # generate answer with history
    message = [
        SystemMessage(content="You are a helpful assistant that answer the question based on provided document"),
        *chat_history,
        HumanMessage(content=f"User question: {combined_input}")
    ]
    result = model.invoke(message)
    answer = result.content
    print("--Generated Answer--")
    print(f"\nAnswer:\n{answer}\n")
    # update chat history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))


def start_chat():
    print("Welcome to the History-Aware RAG Chat!")
    print("Type 'exit' to end the chat.")
    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            print("Ending the chat. Goodbye!")
            break
        ask_question(user_question)


if __name__ == '__main__':
    start_chat()
