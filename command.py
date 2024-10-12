import streamlit as st
import subprocess

# Function to execute Linux commands
def execute_command(command):
    try:
        # Run the command on the shell
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error executing command: {str(e)}"

# Streamlit chatbot interface
def main():
    st.title("Linux Command Execution Chatbot")

    st.write("Enter a Linux command below and the bot will execute it and return the output.")

    # User input for the Linux command
    user_input = st.text_input("Enter Linux command")

    if st.button("Run Command"):
        if user_input:
            # Execute the command and display the output
            output = execute_command(user_input)
            st.text_area("Output", output)
        else:
            st.warning("Please enter a command to execute.")

if __name__ == "__main__":
    main()