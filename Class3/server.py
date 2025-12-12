from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
import tempfile

model = whisper.load_model("base")  # or tiny, small, medium, large
app = FastAPI()

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    # Save uploaded file to temp name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    #audio_bytes = await file.read()
    # TODO: ASR → LLM → TTS
    #return FileResponse("response.wav", media_type="audio/wav")
    result = model.transcribe(tmp_path)
    return {"text": result["text"]} 