import sys
import whisper
import speech_recognition as sr
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit

# Load Whisper model
model = whisper.load_model("small")

class VoiceBotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Bot")
        self.setGeometry(200, 200, 400, 300)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

        self.record_button = QPushButton("Record & Transcribe")
        self.record_button.clicked.connect(self.record_audio)
        self.layout.addWidget(self.record_button)

    def record_audio(self):
        self.text_area.append("Listening...")
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source, phrase_time_limit=5)
        self.text_area.append("Transcribing...")
        # Save audio to file
        with open("temp.wav", "wb") as f:
            f.write(audio.get_wav_data())
        # Whisper transcription
        result = model.transcribe("temp.wav")
        user_text = result["text"]
        self.text_area.append(f"You said: {user_text}")
        # Simple bot response
        bot_reply = self.simple_bot(user_text)
        self.text_area.append(f"Bot: {bot_reply}")

    def simple_bot(self, text):
        text = text.lower()
        if "hello" in text:
            return "Hi there!"
        elif "how are you" in text:
            return "I'm a bot, I feel nothing, but thanks for asking!"
        else:
            return "I don't understand, try saying hello."

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceBotApp()
    window.show()
    sys.exit(app.exec_())
