# online/server.py

import os
import sys
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root (voice-chatbot) is on sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Pipeline imports
from online.stt.whisper_stt import transcribe
from online.retrieval.retriever import get_relevant_chunks
from online.llm.inference import generate_answer
from online.tts.tts_service import synthesize

# Initialize FastAPI app
title = "Voice Chatbot API"
description = "Voice-enabled RAG chatbot with STT, retrieval, LLM, and TTS"
app = FastAPI(title=title, description=description, version="1.0")

# Static mounts:
#  - /static serves Avatar/*.mp4 (waiting/speaking)
#  - /audio  serves generated .wav
static_dir = os.path.join(project_root, "Avatar")
audio_dir = os.path.join(project_root, "online", "temp", "audio")

# create audio directory if missing
os.makedirs(audio_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

# Serve SPA index.html at root
@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(project_root, "index.html"))

# RAG endpoint
@app.post("/ask/")
async def ask(audio: UploadFile = File(...)):
    # 1) save upload
    uid = str(uuid.uuid4())
    in_path = os.path.join(audio_dir, f"{uid}_in.wav")
    out_path = os.path.join(audio_dir, f"{uid}_out.wav")
    with open(in_path, "wb") as f:
        f.write(await audio.read())

    # 2) STT -> text
    query = transcribe(in_path)

    # 3) retrieval
    chunks = get_relevant_chunks(query, top_k=3)

    # 4) LLM answer
    answer = generate_answer(chunks, query)

    # 5) TTS -> wav
    synthesize(answer, out_path)

    # 6) return payload
    return {
        "transcript": query,
        "answer": answer,
        "audio_url": f"/audio/{uid}_out.wav",
        "avatar_waiting": "/static/avatar waiting.mp4",
        "avatar_speaking": "/static/avatar talking.mp4"
    }
