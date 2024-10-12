from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
import os

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

# Run chatbot loop
print("#####################CHATBOT#########################\n")
while input("Type 'c' to continue or 'exit' to exit: ").lower() != "exit":
    question = input("Enter your query: ")
    ans = chain.invoke({"question": question})
    print(ans)