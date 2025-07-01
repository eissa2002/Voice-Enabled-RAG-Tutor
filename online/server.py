import os
import sys
import uuid
import json
import subprocess
import logging
import shutil
from fastapi import FastAPI, File, UploadFile, Request
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

# ─ Configure logging ─
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Voice Chatbot API",
    description="Voice-enabled RAG chatbot with STT, retrieval, LLM, and TTS",
    version="1.0",
)

# ─ Ensure audio temp dir exists ─
audio_dir = os.path.join(project_root, "online", "temp", "audio")
os.makedirs(audio_dir, exist_ok=True)

# ─ Locate FFmpeg ┕
ffmpeg_bin = shutil.which("ffmpeg") or r"C:\Users\eissa.abbas\Desktop\work\work projects\FFmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# ─ Typing simulation helper ─
def make_typing_sim(answer: str):
    arr, cur = [], ""
    for c in answer:
        cur += c
        arr.append(cur)
    return arr

# ─ Startup event to verify FFmpeg ─
@app.on_event("startup")
def verify_ffmpeg():
    if not ffmpeg_bin or not os.path.isfile(ffmpeg_bin):
        logger.error(f"FFmpeg not found at: {ffmpeg_bin}")
        return
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-version"],
            capture_output=True, text=True, check=True
        )
        first_line = proc.stdout.splitlines()[0]
        logger.info(f"FFmpeg found: {first_line}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg test failed:\n{e.stderr}")

@app.get("/", response_class=FileResponse)
async def serve_index():
    idx = os.path.join(project_root, "index.html")
    if not os.path.exists(idx):
        return {"error": "index.html not found at " + idx}
    return FileResponse(idx)

# ─ Greetings / TTS shortcuts ─
GREETINGS = [
    "hello","hi","hey","good morning","good evening","good afternoon","how are you",
    "السلام عليكم","مرحبا","صباح الخير","مساء الخير","أهلا","أهلا وسهلا","كيف حالك","كيف حالكم",
    "salut","bonjour","hola","ciao","hallo","привет","こんにちは","안녕하세요","你好","shalom"
]
GREETINGS_RESPONSES = [
    "Hello! How can I help you today?",
    "Hi there! What would you like to learn?",
    "Hey! Ask me anything from your material.",
    "Welcome! How can I assist you?"
]
def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    return any(t.startswith(g) for g in GREETINGS)

# ─── /chat/ endpoint (text) ───
@app.post("/chat/")
async def chat(request: Request):
    form = await request.form()
    question = form.get("question", "").strip()
    history_raw = form.get("history", "[]")
    try:
        chat_history = json.loads(history_raw)
    except:
        chat_history = []

    # Fast-path greeting
    if is_greeting(question):
        import random
        answer = random.choice(GREETINGS_RESPONSES)
        uid = uuid.uuid4().hex
        out_wav = os.path.join(audio_dir, f"{uid}_out.wav")
        synthesize(answer, out_wav)
        return {
            "transcript": question,
            "answer": answer,
            "citation": "",
            "audio_url": f"/audio/{uid}_out.wav",
            "avatar_waiting": "/static/avatar waiting.mp4",
            "avatar_speaking": "/static/avatar talking.mp4",
            "typing_simulation": make_typing_sim(answer),
        }

    # Full RAG pipeline
    chunks = get_relevant_chunks(question, top_k=3)
    answer, citation = "Sorry, I don’t know.", ""
    if chunks:
        result = generate_answer(chunks, question, chat_history)
        if isinstance(result, tuple):
            answer, citation = result
        else:
            answer = result

    uid = uuid.uuid4().hex
    out_wav = os.path.join(audio_dir, f"{uid}_out.wav")
    synthesize(answer, out_wav)

    return {
        "transcript": question,
        "answer": answer,
        "citation": citation,
        "audio_url": f"/audio/{uid}_out.wav",
        "avatar_waiting": "/static/avatar waiting.mp4",
        "avatar_speaking": "/static/avatar talking.mp4",
        "typing_simulation": make_typing_sim(answer),
    }

# ─── /transcribe/ endpoint (just STT) ───
@app.post("/transcribe/")
async def transcribe_audio(audio: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    in_webm = os.path.join(audio_dir, f"{uid}_stt_in.webm")
    in_wav  = os.path.join(audio_dir, f"{uid}_stt_in.wav")

    # Save upload
    with open(in_webm, "wb") as f:
        f.write(await audio.read())

    # Convert
    audio_path = in_webm
    if ffmpeg_bin and os.path.isfile(ffmpeg_bin):
        try:
            proc = subprocess.run(
                [ffmpeg_bin, "-y", "-i", in_webm, "-ac", "1", "-ar", "16000", in_wav],
                capture_output=True, text=True, check=True
            )
            if os.path.isfile(in_wav) and os.path.getsize(in_wav) > 0:
                audio_path = in_wav
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg STT conversion failed:\n{e.stderr}")

    # Transcribe
    try:
        transcript = transcribe(audio_path)
    except Exception as e:
        logger.error(f"STT failed: {e}")
        transcript = ""

    return {"transcript": transcript}

# ─── /ask/ endpoint (voice → STT+RAG+TTS) ───
@app.post("/ask/")
async def ask(
    audio: UploadFile = File(...),
    request: Request = None,
):
    form = await request.form()
    history_raw = form.get("history", "[]")
    try:
        chat_history = json.loads(history_raw)
    except:
        chat_history = []

    uid     = uuid.uuid4().hex
    in_webm = os.path.join(audio_dir, f"{uid}_in.webm")
    in_wav  = os.path.join(audio_dir, f"{uid}_in.wav")
    out_wav = os.path.join(audio_dir, f"{uid}_out.wav")

    # Save upload
    with open(in_webm, "wb") as f:
        f.write(await audio.read())

    # Convert
    audio_path = in_webm
    if ffmpeg_bin and os.path.isfile(ffmpeg_bin):
        try:
            proc = subprocess.run([
                ffmpeg_bin, "-y", "-i", in_webm, "-ac", "1", "-ar", "16000", in_wav
            ], capture_output=True, text=True, check=True)
            if os.path.isfile(in_wav) and os.path.getsize(in_wav) > 0:
                audio_path = in_wav
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed:\n{e.stderr}")

    # STT
    try:
        query = transcribe(audio_path)
    except Exception as e:
        logger.error(f"STT failed: {e}")
        query = ""
    logger.info(f"[STT] Transcript: {query!r}")

    # Decide response
    if not query.strip():
        answer = "Sorry, I couldn't understand the question."
        citation = ""
    elif is_greeting(query):
        import random
        answer = random.choice(GREETINGS_RESPONSES)
        citation = ""
    else:
        chunks = get_relevant_chunks(query, top_k=3)
        if chunks:
            result = generate_answer(chunks, query, chat_history)
            if isinstance(result, tuple):
                answer, citation = result
            else:
                answer, citation = result, ""
        else:
            answer, citation = "Sorry, I don’t know.", ""

    synthesize(answer, out_wav)
    return {
        "transcript": query,
        "answer": answer,
        "citation": citation,
        "audio_url": f"/audio/{uid}_out.wav",
        "avatar_waiting": "/static/avatar waiting.mp4",
        "avatar_speaking": "/static/avatar talking.mp4",
        "typing_simulation": make_typing_sim(answer),
    }

# ─── Static mounts ───
app.mount("/static", StaticFiles(directory=os.path.join(project_root, "Avatar")), name="static")
app.mount("/audio",  StaticFiles(directory=audio_dir),                         name="audio")