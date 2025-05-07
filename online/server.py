import os
import sys
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ─ Make the voice-chatbot root importable ─
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# ─ Pipeline imports ─
from online.stt.whisper_stt     import transcribe
from online.retrieval.retriever import get_relevant_chunks
from online.llm.inference       import generate_answer
from online.tts.tts_service     import synthesize

app = FastAPI(
    title="Voice Chatbot API",
    description="Voice-enabled RAG chatbot with STT, retrieval, LLM, and TTS",
    version="1.0",
)

# ─ Ensure audio temp dir exists ─
audio_dir = os.path.join(project_root, "online", "temp", "audio")
os.makedirs(audio_dir, exist_ok=True)

# ─── Frontend ───
@app.get("/", response_class=FileResponse)
async def serve_index():
    index_path = os.path.join(project_root, "index.html")
    if not os.path.exists(index_path):
        # helps you debug if index.html isn't where you think
        return {"error": "index.html not found at " + index_path}
    return FileResponse(index_path)

# ─── Voice endpoint ───
@app.post("/ask/")
async def ask(audio: UploadFile = File(...)):
    uid     = str(uuid.uuid4())
    in_wav  = os.path.join(audio_dir, f"{uid}_in.wav")
    out_wav = os.path.join(audio_dir, f"{uid}_out.wav")

    # 1) Save upload
    with open(in_wav, "wb") as f:
        f.write(await audio.read())

    # 2) STT
    query = transcribe(in_wav)

    # 3) Retrieve
    chunks = get_relevant_chunks(query, top_k=3)

    # 4) LLM
    result = generate_answer(chunks, query)
    if isinstance(result, tuple):
        answer, citation = result
    else:
        answer, citation = result, ""

    # 5) TTS
    synthesize(answer, out_wav)

    return {
        "transcript":      query,
        "answer":          answer,
        "citation":        citation,
        "audio_url":       f"/audio/{uid}_out.wav",
        "avatar_waiting":  "/static/avatar waiting.mp4",
        "avatar_speaking": "/static/avatar talking.mp4",
    }

# ─── Text chat endpoint ───
@app.post("/chat/")
async def chat(question: str = Form(...)):
    chunks = get_relevant_chunks(question, top_k=3)
    result = generate_answer(chunks, question)
    if isinstance(result, tuple):
        answer, citation = result
    else:
        answer, citation = result, ""

    uid     = str(uuid.uuid4())
    out_wav = os.path.join(audio_dir, f"{uid}_out.wav")
    synthesize(answer, out_wav)

    return {
        "transcript":      question,
        "answer":          answer,
        "citation":        citation,
        "audio_url":       f"/audio/{uid}_out.wav",
        "avatar_waiting":  "/static/avatar waiting.mp4",
        "avatar_speaking": "/static/avatar talking.mp4",
    }

# ─── Static assets ───
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(project_root, "Avatar")),
    name="static",
)
app.mount(
    "/audio",
    StaticFiles(directory=audio_dir),
    name="audio",
)
