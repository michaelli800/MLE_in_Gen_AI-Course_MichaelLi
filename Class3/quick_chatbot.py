# quick_chatbot.py

responses = {
    "hi": "Hello! How can I help you?",
    "hello": "Hi there!",
    "how are you": "I'm just code, but I'm doing great!",
    "bye": "Goodbye!"
}

def chatbot():
    print("Chatbot: Hi! Type 'bye' to exit.")
    while True:
        user = input("You: ").lower().strip()
        if user == "bye":
            print("Chatbot: Goodbye!")
            break

        reply = responses.get(user, "I don't understand, but I'm learning!")
        print("Chatbot:", reply)

if __name__ == "__main__":
    chatbot()
