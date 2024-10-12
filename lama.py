import ollama

response = ollama.generate(model='llama2',
prompt='what is a qubit?')
print(response['response'])