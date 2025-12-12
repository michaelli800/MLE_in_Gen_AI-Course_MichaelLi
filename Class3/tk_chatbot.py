import tkinter as tk
from tkinter import scrolledtext

responses = {
    "hi": "Hello! How can I help you?",
    "hello": "Hi there!",
    "how are you": "I'm doing great!",
    "bye": "Goodbye!"
}

def get_response(message):
    message = message.lower().strip()
    return responses.get(message, "I don't understand, but I'm learning!")

def send_message():
    user_msg = entry.get().strip()
    if not user_msg:
        return

    chat_box.insert(tk.END, "You: " + user_msg + "\n")

    bot_msg = get_response(user_msg)
    chat_box.insert(tk.END, "Bot: " + bot_msg + "\n\n")

    entry.delete(0, tk.END)
    chat_box.yview(tk.END)

root = tk.Tk()
root.title("Tkinter Chatbot")
root.geometry("400x500")

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry = tk.Entry(root, width=60)
entry.pack(padx=10, pady=(0, 10), side=tk.LEFT, fill=tk.X, expand=True)

send_btn = tk.Button(root, text="Send", command=send_message)
send_btn.pack(padx=(0, 10), pady=(0, 10), side=tk.RIGHT)

root.mainloop()
