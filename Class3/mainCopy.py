from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
import asyncio
from engine import speech_to_text, get_llm_response, text_to_speech
from session import ChatSession

app = FastAPI()


# Serve the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

user_session = ChatSession(max_turns=5)

# Helper function to stream file content and cleanup after
async def stream_and_cleanup(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "rb") as audio_file:
            # Yield chunks of the file
            while chunk := audio_file.read(1024 * 1):  # 8KB chunks
                yield chunk
        # Cleanup output file after streaming is done
        os.remove(file_path)

@app.get("/")
async def read_index():
    # This sends your HTML file when you visit http://localhost:8000/
    return FileResponse('static/index.html')

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    input_path = f"temp_in_{session_id}.wav"
    output_path = f"temp_out_{session_id}.mp3"

    # 1. Save uploaded audio
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. ASR: Speech -> Text (Keep this synchronous or wrap in thread)
    user_text = speech_to_text(input_path)
    print("text from audio:" + user_text)
    user_session.add_message("user", user_text)

    # 3. LLM: Get AI Response ; Process with 5-turn memory
    context = user_session.get_context()
    ai_response = get_llm_response(context)
    user_session.add_message("assistant", ai_response)

    # 4. TTS: Text -> Speech ; Generate the output file
	# Note: If your TTS engine supports streaming directly (like OpenAI), 
    # you could yield directly from the engine here.
    
    text_to_speech(ai_response, output_path)

    # Cleanup input file immediately
    os.remove(input_path)

    # 5. Return a Streaming Response
    return StreamingResponse(
        stream_and_cleanup(output_path), 
        media_type="audio/mpeg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)