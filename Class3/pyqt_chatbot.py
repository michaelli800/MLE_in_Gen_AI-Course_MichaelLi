import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton
)

responses = {
    "hi": "Hello! How can I help you?",
    "hello": "Hi there!",
    "how are you": "I'm doing great!",
    "bye": "Goodbye!"
}

def get_response(message):
    message = message.lower().strip()
    return responses.get(message, "I don't understand, but I'm learning!")

class ChatbotUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Chatbot")
        self.resize(450, 550)

        layout = QVBoxLayout()

        self.chat_box = QTextEdit()
        self.chat_box.setReadOnly(True)
        layout.addWidget(self.chat_box)

        self.entry = QLineEdit()
        self.entry.returnPressed.connect(self.send_message)
        layout.addWidget(self.entry)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        layout.addWidget(self.send_btn)

        self.setLayout(layout)

    def send_message(self):
        user_msg = self.entry.text().strip()
        if not user_msg:
            return

        self.chat_box.append(f"<b>You:</b> {user_msg}")

        bot_msg = get_response(user_msg)
        self.chat_box.append(f"<b>Bot:</b> {bot_msg}<br>")

        self.entry.clear()

app = QApplication(sys.argv)
window = ChatbotUI()
window.show()
sys.exit(app.exec_())
