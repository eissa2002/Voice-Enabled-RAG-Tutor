# online/server.py

import os
import sys
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Make the voice-chatbot root importable
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, project_root)

# Pipeline imports
from online.stt.whisper_stt import transcribe
from online.retrieval.retriever import get_relevant_chunks
from online.llm.inference import generate_answer
from online.tts.tts_service import synthesize

# Initialize FastAPI
app = FastAPI(
    title="Voice Chatbot API",
    description="Voice-enabled RAG chatbot with STT, retrieval, LLM, and TTS",
    version="1.0",
)

# Ensure audio temp directory exists
audio_dir = os.path.join(project_root, "online", "temp", "audio")
os.makedirs(audio_dir, exist_ok=True)

@app.post("/ask/")
async def ask(audio: UploadFile = File(...)):
    # Save uploaded audio
    uid = str(uuid.uuid4())
    in_path = os.path.join(audio_dir, f"{uid}_in.wav")
    out_path = os.path.join(audio_dir, f"{uid}_out.wav")
    with open(in_path, "wb") as f:
        f.write(await audio.read())

    # Pipeline: STT → retrieve → LLM → TTS
    query = transcribe(in_path)
    chunks = get_relevant_chunks(query, top_k=3)
    answer = generate_answer(chunks, query)
    synthesize(answer, out_path)

    return {
        "transcript": query,
        "answer": answer,
        "audio_url": f"/audio/{uid}_out.wav",
        "avatar_waiting": "/static/avatar waiting.mp4",
        "avatar_speaking": "/static/avatar talking.mp4",
    }

# Mount avatars (MP4s)
avatar_dir = os.path.join(project_root, "Avatar")
app.mount(
    "/static",
    StaticFiles(directory=avatar_dir),
    name="avatar",
)

# Mount generated audio
app.mount(
    "/audio",
    StaticFiles(directory=audio_dir),
    name="audio",
)

# Serve the frontend (index.html + JS/CSS) from project root
app.mount(
    "/",
    StaticFiles(directory=project_root, html=True),
    name="frontend",
)
