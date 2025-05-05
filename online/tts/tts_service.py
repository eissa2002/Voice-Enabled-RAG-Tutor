# online/tts/tts.py

import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()
# You can adjust properties like rate or volume if desired:
engine.setProperty('rate', 150)  # speech rate (words per minute)
engine.setProperty('volume', 1.0)  # volume: min=0.0, max=1.0


def synthesize(text: str, output_path: str) -> None:
    """
    Convert the given text to speech and save to an audio file (e.g., WAV).

    :param text: The text string to speak.
    :param output_path: Path to write the audio file (e.g., "response.wav").
    """
    # Queue the text to be spoken to file
    engine.save_to_file(text, output_path)
    # Run the speech engine
    engine.runAndWait()


if __name__ == "__main__":
    # Simple test
    sample = "Hello, this is a test of the text to speech engine."
    out = "test_output.wav"
    synthesize(sample, out)
    print(f"Saved speech to {out}")
