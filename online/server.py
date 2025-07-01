import os
import sys
import uuid
import json
import subprocess
import logging
import shutil
import difflib
import re

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ─ Make project root importable ─
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# ─ Pipeline imports ─
from online.stt.whisper_stt import transcribe
from online.retrieval.retriever import get_relevant_chunks
from online.llm.inference import generate_answer, llm
from online.tts.tts_service import synthesize

# ─ Logging ─
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Voice Chatbot API",
    description="Voice-enabled RAG tutor with automatic bilingual support",
    version="1.0",
)

# ─ Ensure audio temp dir exists ─
audio_dir = os.path.join(project_root, "online", "temp", "audio")
os.makedirs(audio_dir, exist_ok=True)

# ─ Locate FFmpeg ─
ffmpeg_bin = shutil.which("ffmpeg") or r"C:\Users\eissa.abbas\Desktop\work\work projects\FFmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# ─ Typing simulation helper ─
def make_typing_simulation(answer_text: str):
    sim = []
    cur = ""
    for c in answer_text:
        cur += c
        sim.append(cur)
    return sim

# ─ Greetings / fuzzy match ─
GREETINGS = [
    "hello","hi","hey","good morning","good evening","good afternoon","how are you",
    "السلام عليكم","مرحبا","صباح الخير","مساء الخير","أهلا","أهلا وسهلا","كيف حالك","كيف حالكم"
]
GREETINGS_RESPONSES_EN = [
    "Hello! How can I help you today?",
    "Hi there! What would you like to learn?",
    "Hey! Ask me anything from your material.",
    "Welcome! How can I assist you?"
]
GREETINGS_RESPONSES_AR = [
    "مرحباً! كيف يمكنني مساعدتك اليوم؟",
    "أهلاً! بماذا تحب أن تتعلم؟",
    "مرحباً! اسألني أي شيء من موادك.",
    "أهلاً وسهلاً! كيف أستطيع مساعدتك؟"
]
G_LOWER = [g.lower() for g in GREETINGS]

def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    for g in G_LOWER:
        if t == g or t.startswith(g):
            return True
    # fuzzy on whole
    if difflib.get_close_matches(t, G_LOWER, n=1, cutoff=0.8):
        return True
    # fuzzy on first word
    first = t.split()[0] if t.split() else ""
    if difflib.get_close_matches(first, G_LOWER, n=1, cutoff=0.8):
        return True
    return False

# ─ Detect language by presence of Arabic chars ─
def detect_language(text: str) -> str:
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    return "en"

# ─ Verify FFmpeg on startup ─
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
        logger.info(f"FFmpeg found: {proc.stdout.splitlines()[0]}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg test failed:\n{e.stderr}")

# ─ Serve index.html ─
@app.get("/", response_class=FileResponse)
async def serve_index():
    path = os.path.join(project_root, "index.html")
    if not os.path.exists(path):
        return {"error": f"index.html not found at {path}"}
    return FileResponse(path)

# ─── /chat/ endpoint ───
@app.post("/chat/")
async def chat(request: Request):
    form = await request.form()
    question = form.get("question", "").strip()
    history_raw = form.get("history", "[]")
    try:
        chat_history = json.loads(history_raw)
    except json.JSONDecodeError:
        chat_history = []

    lang = detect_language(question)

    # greeting
    if is_greeting(question):
        import random
        if lang == "ar":
            answer = random.choice(GREETINGS_RESPONSES_AR)
        else:
            answer = random.choice(GREETINGS_RESPONSES_EN)
        citation = ""
    else:
        # retrieval + LLM
        chunks = get_relevant_chunks(question, top_k=3)
        if chunks:
            answer, citation = generate_answer(
                chunks,
                question,
                chat_history,
                target_lang=lang
            )
        else:
            answer = "Sorry, I don’t know." if lang=="en" else "عذراً، لا أعرف."
            citation = ""

    # TTS
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
        "typing_simulation": make_typing_simulation(answer),
    }

# ─── /transcribe/ endpoint ───
@app.post("/transcribe/")
async def transcribe_audio(audio: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    in_webm = os.path.join(audio_dir, f"{uid}_in.webm")
    in_wav  = os.path.join(audio_dir, f"{uid}_in.wav")
    with open(in_webm, "wb") as f:
        f.write(await audio.read())

    audio_path = in_webm
    if ffmpeg_bin and os.path.isfile(ffmpeg_bin):
        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", in_webm, "-ac", "1", "-ar", "16000", in_wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            if os.path.isfile(in_wav) and os.path.getsize(in_wav) > 0:
                audio_path = in_wav
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg STT conversion failed:\n{e.stderr}")

    try:
        transcript = transcribe(audio_path)
    except Exception as e:
        logger.error(f"STT failed: {e}")
        transcript = ""

    return {"transcript": transcript}

# ─── /ask/ endpoint ───
@app.post("/ask/")
async def ask(audio: UploadFile = File(...), request: Request = None):
    form = await request.form()
    history_raw = form.get("history", "[]")
    try:
        chat_history = json.loads(history_raw)
    except json.JSONDecodeError:
        chat_history = []

    uid = uuid.uuid4().hex
    in_webm = os.path.join(audio_dir, f"{uid}_in.webm")
    in_wav  = os.path.join(audio_dir, f"{uid}_in.wav")
    out_wav = os.path.join(audio_dir, f"{uid}_out.wav")
    with open(in_webm, "wb") as f:
        f.write(await audio.read())

    audio_path = in_webm
    if ffmpeg_bin and os.path.isfile(ffmpeg_bin):
        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", in_webm, "-ac", "1", "-ar", "16000", in_wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            if os.path.isfile(in_wav) and os.path.getsize(in_wav) > 0:
                audio_path = in_wav
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed:\n{e.stderr}")

    try:
        question = transcribe(audio_path)
    except Exception as e:
        logger.error(f"STT failed: {e}")
        question = ""
    logger.info(f"[STT] Transcript: {question!r}")

    lang = detect_language(question)

    if not question.strip():
        answer = "Sorry, I couldn't understand the question." if lang=="en" else "عذراً، لم أتمكن من الفهم."
        citation = ""
    elif is_greeting(question):
        import random
        answer = random.choice(GREETINGS_RESPONSES_AR if lang=="ar" else GREETINGS_RESPONSES_EN)
        citation = ""
    else:
        chunks = get_relevant_chunks(question, top_k=3)
        if chunks:
            answer, citation = generate_answer(chunks, question, chat_history, target_lang=lang)
        else:
            answer = "Sorry, I don’t know." if lang=="en" else "عذراً، لا أعرف."
            citation = ""

    synthesize(answer, out_wav)

    return {
        "transcript": question,
        "answer": answer,
        "citation": citation,
        "audio_url": f"/audio/{uid}_out.wav",
        "avatar_waiting": "/static/avatar waiting.mp4",
        "avatar_speaking": "/static/avatar talking.mp4",
        "typing_simulation": make_typing_simulation(answer),
    }

# ─── /translate/ endpoint ───
@app.post("/translate/")
async def translate_text(request: Request):
    body = await request.json()
    text = body.get("text", "")
    # detect original
    original_lang = detect_language(text)
    target = "en" if original_lang=="ar" else "ar"

    if target == "ar":
        prompt = (
            f"Translate the following English text into Arabic. "
            f"Only return the translated text, nothing else:\n\n{text}\n\nالترجمة:"
        )
    else:
        prompt = (
            f"Translate the following text into English. "
            f"Only return the translated text, nothing else:\n\n{text}\n\nTranslation:"
        )

    try:
        resp = llm.generate([prompt])
        translation = resp.generations[0][0].text.strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        translation = ""

    return {"translation": translation, "citation": "- Translated by AI"}

# ─── Static mounts ───
app.mount("/static",
          StaticFiles(directory=os.path.join(project_root, "Avatar")),
          name="static")
app.mount("/audio",
          StaticFiles(directory=audio_dir),
          name="audio")
