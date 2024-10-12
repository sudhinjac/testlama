from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
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
'''
# Local PDF file uploads
if local_path:
  loader = UnstructuredPDFLoader(file_path=local_path)
  data = loader.load()
else:
  print("Upload a PDF file")
  '''
# Preview first page
#print(data[0].page_content)
# Split and chunk 
for filename in os.listdir(local_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(local_path, filename)
            loader = UnstructuredPDFLoader(file_path=file_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#chunks = text_splitter.split_documents(data)
embedding_function = OllamaEmbeddings(model="nomic-embed-text",show_progress=True)
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
    collection_name="local-rag"
)
Chroma.from_documents(chunks, embedding_function, persist_directory=vector_db_path)
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

#chain.invoke(input(""))
print("#####################CHATBOT#########################\n")
while input("Type c to continue or exit to exit: ")!="exit":
    question = input("Enter your query: ")
    ans = chain.invoke(question)
    print(ans)