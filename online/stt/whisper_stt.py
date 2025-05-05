# online/stt/whisper_stt.py

from faster_whisper import WhisperModel

# Initialize the Faster Whisper model once
model = WhisperModel(
    model_size_or_path="small",  # choose tiny/small/medium/large-v2
    device="cpu",                # or "cuda" if you have a GPU
    compute_type="int8"          # reduces memory footprint
)

def transcribe(audio_path: str) -> str:
    """
    Transcribe the given audio file to text using faster-whisper.
    """
    segments, _ = model.transcribe(audio_path)
    # Concatenate segment texts
    return "".join([segment.text for segment in segments])
