from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
import asyncio
from engine import speech_to_text, get_audio_stream
from session import ChatSession

app = FastAPI()


# Serve the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

user_session = ChatSession(max_turns=5)

@app.get("/")
async def read_index():
    # This sends your HTML file when you visit http://localhost:8000/
    return FileResponse('static/index.html')

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    input_path = f"temp_in_{session_id}.wav"

    # 1. Save uploaded audio
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Transcribe (ASR)
    user_text = speech_to_text(input_path)
    print("text from audio:" + user_text)
    user_session.add_message("user", user_text)
    
    # Clean up input file immediately
    os.remove(input_path)

    # 3. Return the Stream directly from the engine
    # We pass the conversation context to our new streaming function
    return StreamingResponse(
        get_audio_stream(user_session.get_context()), 
        media_type="audio/mpeg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)