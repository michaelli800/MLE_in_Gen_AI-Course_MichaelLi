import ollama

def get_completion(prompt, model="llama3.2"):
    messages = [{"role": "user", "content": prompt}]
    response = ollama.chat(
        model=model,
        messages=messages#,
        #options={"temperature": 0}
    )
    return response["message"]["content"]

#prompt = "What is the capital of France?"
#text = """ Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. It has applications in various fields, including healthcare, finance, and transportation."""
#prompt = f"Summarize the following text:\n{text}"

# text = """John Doe, a 29-year-old software engineer from San Francisco, recently joined OpenAI as a research scientist."""
# prompt = f"Extract the name and occupation from the following text:\n{text}"

# text = "The weather is nice today."
# prompt = f"Translate the following text to French:\n{text}"

#prompt = "Write a short story about a dragon who learns to code."
#prompt = "As a professional chef, explain how to make a perfect omelette."

# prompt = """
# Translate the following English phrases to French:

# English: Hello
# French: Bonjour

# English: Thank you
# French: Merci

# English: Good night
# French:
# """

#prompt = "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Explain your reasoning."

#response = get_completion(prompt)
#print(response)

def get_completion_with_system_prompt(system_prompt, user_prompt, model="llama3.2"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]

# Example usage
# system_prompt = "You are a helpful assistant that provides concise and accurate information."
# user_prompt = "Can you explain the importance of data privacy?"

# response = get_completion_with_system_prompt(system_prompt, user_prompt)
# print(response)

#Example usage
system_prompt = "You are a funny assistant that provides humorous information."
user_prompt = "Can you explain the importance of data privacy?"

response = get_completion_with_system_prompt(system_prompt, user_prompt)
print(response)

model="llama3.2"
messages = [
    {"role": "system", "content": "You are a helpful assistant knowledgeable in history."},
    {"role": "user", "content": "Who was the first president of the United States?"},
    {"role": "assistant", "content": "George Washington was the first president of the United States."},
    {"role": "user", "content": "When did he take office?"}
]
response = ollama.chat(model=model, messages=messages)
print(response["message"]["content"])
