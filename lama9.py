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

# Load and process PDF files
chunks = []
for filename in os.listdir(local_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(local_path, filename)
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks.extend(text_splitter.split_documents(data))

# Set up embeddings and vector store
embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_function,
    persist_directory=vector_db_path,
    collection_name="local-rag"
)

# LLM for question generation and answer retrieval
local_model = "llama2"
llm = ChatOllama(model=local_model)

# Query prompt for generating multiple queries
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Set up the retriever with multiple queries
retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(), 
    llm=llm,
    prompt=QUERY_PROMPT
)

# RAG prompt for answering the question
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Chain configuration
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run chatbot loop
print("#####################CHATBOT#########################\n")
while input("Type c to continue or exit to exit: ") != "exit":
    question = input("Enter your query: ")
    ans = chain.invoke({"context": retriever, "question": question})
    print(ans)