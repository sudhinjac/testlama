import ollama

print(ollama.chat(model='llama3:instruct', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]))