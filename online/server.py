# online/server.py

import os
import sys
import uuid
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ─ Make the voice-chatbot root importable ─
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# ─ Pipeline imports ─
from online.stt.whisper_stt import transcribe
from online.retrieval.retriever import get_relevant_chunks
from online.llm.inference import generate_answer
from online.tts.tts_service import synthesize

app = FastAPI(
    title="Voice Chatbot API",
    description="Voice-enabled RAG chatbot with STT, retrieval, LLM, and TTS",
    version="1.0",
)

# ─ Ensure audio temp directory exists ─
audio_dir = os.path.join(project_root, "online", "temp", "audio")
os.makedirs(audio_dir, exist_ok=True)

@app.post("/ask/")
async def ask(audio: UploadFile = File(...)):
    # 1) Save raw upload
    uid = str(uuid.uuid4())
    ext = Path(audio.filename or "").suffix or ".wav"
    raw_path = os.path.join(audio_dir, f"{uid}_raw{ext}")
    with open(raw_path, "wb") as f:
        f.write(await audio.read())

    # 2) Convert to 16 kHz mono WAV
    wav_path = os.path.join(audio_dir, f"{uid}.wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", raw_path,
        "-ar", "16000",      # sample rate
        "-ac", "1",          # mono
        wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # (optional) delete raw upload to save space
    try:
        os.remove(raw_path)
    except OSError:
        pass

    # 3) STT → corrected text
    query = transcribe(wav_path)
    print("🔍 Transcribed + corrected text:", query)

    # 4) Retrieval → top-3 chunks
    chunks = get_relevant_chunks(query, top_k=3)

    # 5) LLM → answer (+ optional citation)
    result = generate_answer(chunks, query)
    if isinstance(result, tuple):
        answer, citation = result
    else:
        answer, citation = result, ""

    # 6) TTS on just the answer
    synthesize(answer, wav_path.replace(".wav", "_out.wav"))

    return {
        "transcript":      query,
        "answer":          answer,
        "citation":        citation,
        "audio_url":       f"/audio/{uid}_out.wav",
        "avatar_waiting":  "/static/avatar waiting.mp4",
        "avatar_speaking": "/static/avatar talking.mp4",
    }

# ─ Mount static assets ─

# 7a) Avatars (mp4s)
avatar_dir = os.path.join(project_root, "Avatar")
app.mount("/static", StaticFiles(directory=avatar_dir), name="avatar")

# 7b) Generated audio
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

# 7c) Frontend (index.html + JS/CSS) from project root
app.mount("/", StaticFiles(directory=project_root, html=True), name="frontend")
