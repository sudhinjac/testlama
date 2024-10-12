import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Paths
vector_db_path = "/mnt/c/testlama/vector_db"

# Load preprocessed vector store and ensure embeddings are set
embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
vector_db = Chroma(persist_directory=vector_db_path, collection_name="local-rag", embedding_function=embedding_function)

# LLM for question answering
local_model = "llama2"
llm = ChatOllama(model=local_model)

# Simple retriever
retriever = vector_db.as_retriever()

# Direct retrieval prompt for answering the question
template = """Use the following context to answer the question:
{context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def retrieve_context(question):
    """Retrieve relevant documents from the vector store."""
    documents = retriever.get_relevant_documents(question)
    # Combine document texts into a single context string
    context = "\n".join(doc.page_content for doc in documents)
    return context

@st.cache_data
def get_answer(question):
    """Generate an answer based on the retrieved context."""
    context = retrieve_context(question)
    answer = chain.invoke({"context": context, "question": question})
    return answer

# Chain configuration
chain = (
    {"context": retrieve_context, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit app
st.title("PDF Q&A Chatbot")
st.write("Ask questions based on preloaded PDF documents!")

# User input for questions
question = st.text_input("Enter your query:")

if question:
    try:
        # Display the asked question
        st.write("### Your Question:")
        st.write(question)
        
        # Fetch and display the answer
        ans = get_answer(question)
        st.write("### Answer:")
        st.write(ans)
    except Exception as e:
        st.write("An error occurred:", e)