# online/stt/record_test.py

import os
import sys
import threading

import numpy as np
import sounddevice as sd
import wavio

# ─ Adjust Python path so we can import `online` as a package ─
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────────

from online.stt.whisper_stt import transcribe

def record_until_enter(fs: int = 16000, channels: int = 1, out_file: str = "mic_test.wav"):
    stop_event = threading.Event()
    frames = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"⚠️  {status}", file=sys.stderr)
        frames.append(indata.copy())

    def wait_for_enter():
        input("➤ Press Enter to stop recording...\n")
        stop_event.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    print(f"⏺️  Recording... (sample rate={fs}Hz, channels={channels})")
    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        while not stop_event.is_set():
            sd.sleep(100)

    audio_data = np.concatenate(frames, axis=0)
    wavio.write(out_file, audio_data, fs, sampwidth=2)
    print(f"💾 Saved recording to {out_file}\n")
    return out_file

if __name__ == "__main__":
    # 1) Record until you hit Enter
    wav_path = record_until_enter()

    # 2) Transcribe (English-only)
    print("\n📝 Transcribing with Whisper (English-only)…")
    text = transcribe(wav_path)
    print("\n🗣️  Transcript:\n", text)
