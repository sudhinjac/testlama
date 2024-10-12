import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
import os

# Paths
local_path = "/mnt/c/testlama/PDF"
vector_db_path = "/mnt/c/testlama/vector_db"

# Streamlit app configuration
st.title("PDF Q&A Chatbot")
st.write("Upload your PDF files and ask questions!")

# File uploader
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Load and process PDF files
    chunks = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()

        # Use a refined chunk size with overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks.extend(text_splitter.split_documents(data))

    # Set up embeddings and vector store
    embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory=vector_db_path,
        collection_name="local-rag"
    )

    # LLM for question answering
    local_model = "llama2"
    llm = ChatOllama(model=local_model)

    # Simple retriever without multi-query overhead
    retriever = vector_db.as_retriever()

    # Direct retrieval prompt for answering the question
    template = """Use the following context to answer the question:
    {context}
    Question: {question}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Chain configuration
    chain = (
        {"context": lambda inputs: retriever.get_relevant_documents(inputs['question']), "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    # User input for questions
    question = st.text_input("Enter your query:")

    if question:
        ans = chain.invoke({"question": question})
        st.write("### Answer:")
        st.write(ans)