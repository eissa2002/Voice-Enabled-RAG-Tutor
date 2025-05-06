# online/stt/whisper_stt.py

from faster_whisper import WhisperModel
import sys


# Initialize the Faster Whisper model once
model = WhisperModel(
    model_size_or_path="medium",  # tiny/small/medium/large-v2...
    device="cpu",                 # or "cuda"
    compute_type="int8"           # reduces memory
)

def transcribe(audio_path: str) -> str:
    """
    Transcribe the given audio file, forcing English only.
    """
    segments, _ = model.transcribe(
        audio_path,
        language="en"   # <<< force English
    )
    return "".join(segment.text for segment in segments)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_stt.py <path/to/audio.wav>")
        sys.exit(1)
    wav = sys.argv[1]
    print(f"\nTranscribing ➜ {wav}\n")
    try:
        result = transcribe(wav)
        print("➡️  Result:\n", result)
    except Exception as e:
        print("❌  Transcription failed:", e)
