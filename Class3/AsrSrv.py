from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    # TODO: ASR → LLM → TTS
    user_text = transcribe_audio(audio_bytes)
    bot_text = generate_response(user_text)
    return {"text": bot_text}
    #return FileResponse("response.wav", media_type="audio/wav")

import whisper

asr_model = whisper.load_model("small")

def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav")
    return result["text"]

from transformers import pipeline
#llm = pipeline("text-generation", model="meta-llama/Llama-3-8B")
llm = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

conversation_history = []

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    # Construct prompt from history
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"
    outputs = llm(prompt, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response

