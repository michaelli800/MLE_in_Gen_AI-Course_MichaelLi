from collections import deque

class ChatSession:
    def __init__(self, max_turns=5):
        # 5 turns = 10 messages (User + Assistant)
        self.history = deque(maxlen=max_turns * 2)

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_context(self):
        return list(self.history)