# Voice-Enabled RAG Tutor

A self‑hosted, voice‑enabled Retrieval‑Augmented‑Generation (RAG) tutor that can serve as an expert in *any* domain—from computer vision to NLP, finance, healthcare, and beyond. Speak your question into the mic; the system transcribes it, retrieves relevant document chunks, generates a grounded answer with a local LLM, and returns both text and TTS audio—complete with avatar animations.

---

## 📂 Project Structure

```plaintext
.
├── Avatar/                   # waiting & speaking avatar animations (MP4)
│   ├── avatar waiting.mp4
│   └── avatar talking.mp4
├── config/
│   └── settings.yaml        # optional configuration flags
├── data/
│   ├── raw/                 # put your domain PDFs & PPTX files here
│   └── chunks/              # auto‑generated JSON document chunks
├── db/
│   └── chroma_index/        # persisted Chroma vectorstore
├── offline/
│   ├── loaders.py           # load PDFs + PPTX → LangChain Documents
│   ├── splitter.py          # group pages & split into text chunks
│   └── indexer.py           # embed & persist chunks into Chroma
├── online/
│   ├── stt/
│   │   ├── whisper_stt.py   # faster‑whisper wrapper (English only)
│   │   └── record_test.py   # CLI for mic testing & transcription
│   ├── retrieval/
│   │   └── retriever.py     # load Chroma & similarity search
│   ├── llm/
│   │   └── inference.py     # build prompt, call OllamaLLM, format citations
│   ├── tts/
│   │   └── tts_service.py   # synthesize answer to WAV via TTS engine
│   ├── temp/                # working audio files (in/out)
│   └── server.py            # FastAPI app (endpoints `/ask/` & `/chat/`)
├── index.html               # browser UI (record, display, playback)
├── README.md                # this file
├── requirements.txt         # pinned dependencies
└── test_output.wav          # example recording
```

---

## ⚙️ Installation

1. **Clone & enter**

   ```bash
   git clone <repo_url> && cd voice‑chatbot
   ```

2. **Python env & install**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   .venv\Scripts\activate.bat    # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

> No OCR or Poppler/Tesseract needed—PDFs load directly with `PyPDFLoader`, PPTX via `python-pptx`.

---

## 🛠️ Offline Indexing

1. **Split** your domain materials (PDF + PPTX → JSON chunks):

   ```bash
   python offline/splitter.py
   ```

2. **Embed & persist** chunks into Chroma:

   ```bash
   python offline/indexer.py
   ```

---

## 🚀 Running the API

```bash
uvicorn online.server:app --reload --port 8000
```

* **POST** `/ask/` (audio upload) → returns JSON:

  ```jsonc
  {
    "transcript": "user question text",
    "answer":     "LLM’s grounded answer",
    "citation":   "- file.pdf (page X)\n- other.pdf (page Y)",
    "audio_url":  "/audio/<uid>_out.wav",
    "avatar_waiting":  "/static/avatar waiting.mp4",
    "avatar_speaking": "/static/avatar talking.mp4"
  }
  ```

* **POST** `/chat/` (form text) → pure-text chat without recording.

---

## 🎤 Web UI

Browse to [http://127.0.0.1:8000](http://127.0.0.1:8000):

1. **Record** your question via mic.
2. **Stop** → see **Transcript**, **Answer**, and hear audio playback + avatar animation.

---

## 🔧 Customization

* **Change LLM**: edit `online/llm/inference.py` (e.g. use `OllamaLLM` or other).
* **Chunking**: adjust `pages_per_chunk`, `chunk_size`, `chunk_overlap` in `offline/splitter.py`.
* **Retrieval**: tweak `top_k` or threshold in `online/retrieval/retriever.py`.
* **STT**: swap `model_size_or_path` or `device` in `online/stt/whisper_stt.py`.
* **TTS**: point `synthesize(...)` to your favorite TTS in `online/tts/tts_service.py`.

---

## 📝 Troubleshooting

* **Empty transcription?**

  * Check mic permissions & audio format.
  * Use `python online/stt/record_test.py` to record + transcribe.

* **Missing content in answers?**

  * Ensure `data/raw/` files contain selectable text (not scanned images).
  * Tweak loader in `offline/loaders.py` if needed.

* **Chroma issues?**

  * Delete `db/chroma_index/` and re-run `offline/indexer.py`.

Happy teaching & learning across *any* domain!
— Your Voice‑Enabled RAG Tutor
