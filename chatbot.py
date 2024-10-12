import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import os

local_path = "/mnt/c/testlama/PDF"
vector_db_path = "/mnt/c/testlama/vector_db"

st.title("PDF Query Application")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    chunks = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(local_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        file_chunks = text_splitter.split_documents(data)
        chunks.extend(file_chunks)

    embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        collection_name="local-rag",
        persist_directory=vector_db_path
    )

    # LLM from Ollama
    local_model = "llama2"
    llm = ChatOllama(model=local_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    st.success("Embeddings created and saved successfully!")

    question = st.text_input("Enter your query:")
    if question:
        ans = chain.invoke({"question": question})
        st.write("Answer:", ans)
else:
    st.warning("Please upload at least one PDF file.")