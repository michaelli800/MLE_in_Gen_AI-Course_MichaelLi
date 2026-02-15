import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def speech_to_text(audio_path):
    # Using OpenAI Whisper for high accuracy
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    return transcript.text

def get_llm_response(messages1):
    print (messages1)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        #messages=[{"role": "system", "content": "You are a concise voice assistant."}] + messages1
        messages= messages1
    )
    print(response.choices[0].message)
    return response.choices[0].message.content

def text_to_speech(text, output_path):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(output_path)

