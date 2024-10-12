from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

# Paths
local_path = "/mnt/c/testlama/PDF"
vector_db_path = "/mnt/c/testlama/vector_db"

# Load and process PDF files
chunks = []
for filename in os.listdir(local_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(local_path, filename)
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