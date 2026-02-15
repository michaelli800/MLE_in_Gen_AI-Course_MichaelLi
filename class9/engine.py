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

def get_audio_stream(messages):
    print(messages)
    """
    This generator chains LLM streaming and TTS streaming together.
    """
    # 1. Stream the LLM text response
    # We use a full response here for simplicity, but gpt-4o-mini is fast enough.
    # For maximum speed, you'd stream words from the LLM into the TTS.
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    ai_text = response.choices[0].message.content
    print(f"AI Response: {ai_text}")

    # 2. Stream the TTS audio response
    # Using 'with_streaming_response' returns a context manager for the raw stream
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=ai_text,
        response_format="mp3"
    ) as response:
        # This yields chunks of audio data as they arrive from OpenAI
        for chunk in response.iter_bytes(chunk_size=1024):
            yield chunk