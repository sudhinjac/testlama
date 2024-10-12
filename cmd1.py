import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# LLM for question answering
local_model = "llama3.1"
llm = ChatOllama(model=local_model)

# Function to execute Linux commands
def execute_linux_command(command):
    try:
        output = os.popen(command).read()
        if not output:
            output = "Command executed successfully, but no output was returned."
    except Exception as e:
        output = f"Error executing command: {str(e)}"
    return output

# Streamlit interface
st.title("Chatbot with Linux Command Execution")

# User input
user_input = st.text_input("Enter your question or command:")

# Process the user input
if user_input:
    if user_input.lower().startswith("execute command"):
        # Extract the command to be executed
        command_to_execute = user_input.lower().replace("execute command", "").strip()
        
        # Execute the Linux command and return the output
        if command_to_execute:
            result = execute_linux_command(command_to_execute)
            st.write(f"Command Output:\n{result}")
        else:
            st.write("Please provide a command to execute after 'execute command'.")
    else:
        # Answer general queries using the LLM
        prompt_template = ChatPromptTemplate.from_template("{question}")
        prompt = prompt_template.format_prompt(question=user_input)
        
        # Use LangChain's LLM to process the query
        answer = llm(prompt.to_messages())
        st.write(f"Answer:\n{answer.content}")