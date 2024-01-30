import langchain
from langchain.llms import OpenAI
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Configure components
openai_api_key = "YOUR_OPENAI_API_KEY"
weaviate_url = "http://localhost:8080"

# Create instances
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vstore = Weaviate(
    weaviate_url=weaviate_url,
    collection_name="my-collection",
    embedding=embedding,
)
retriever = vstore.retrieve("text")
llm = OpenAI(temperature=0.7)
memory = ConversationBufferMemory()

# Define a function for handling user queries
def handle_query(query):
    # Retrieve relevant documents
    retrieved_docs = retriever(query)

    # Format retrieved documents for model input
    retrieved_text = "\n".join(doc.text for doc in retrieved_docs)

    # Construct prompt with retrieved text
    prompt = f"{retrieved_text}\nAnswer the question: {query}"

    # Generate response using LLM and memory
    response = llm(prompt, memory=memory)

    # Update conversation memory
    memory.update(query, response)

    return response

# Example usage
query = "What are the benefits of using RAG pipelines?"
response = handle_query(query)
print(response)
