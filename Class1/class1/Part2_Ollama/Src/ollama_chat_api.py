import requests
import json

url = "http://localhost:11434/api/chat"
data = {
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "options": {"temperature": 0},
}

response = requests.post(url, json=data)
print(response.text)
data = response.text
contents = []
for line in data.strip().splitlines():
    obj = json.loads(line)
    msg = obj.get("message", {}).get("content", "")
    contents.append(msg)

# Combine into a single string
full_message = "".join(contents).strip()

print(full_message)
#print(json.loads(response.text)["message"]["content"])