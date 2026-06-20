import os
import time
import subprocess
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def compress_audio(file_path: str) -> str:
    """Compress audio to mono 16kHz mp3 — reduces file from 56MB to ~7MB
    and uses far less of Groq's audio quota."""
    output_path = "compressed_audio.mp3"
    
    print("🗜️ Compressing audio before sending to Groq...")
    subprocess.run([
        "ffmpeg", "-i", file_path,
        "-ar", "16000",  # 16kHz is enough for speech
        "-ac", "1",      # mono
        "-b:a", "32k",   # low bitrate
        output_path,
        "-y"             # overwrite if exists
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Compressed to {new_size:.1f} MB")
    return output_path

def transcribe_audio(file_path: str) -> str:
    """
    Sends audio/video file to Groq's Whisper large-v3
    and returns the Arabic transcript.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.1f} MB")

    if file_size_mb > 25:
        print("⚠️ File is over 25MB — splitting into chunks...")
        return transcribe_large_file(file_path, client)

    print("🎙️ Sending audio to Groq Whisper large-v3...")

    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            language="ar",
            response_format="text"
        )

    print("✅ Transcription complete!")
    return transcription

def transcribe_large_file(file_path: str, client: Groq) -> str:
    from pydub import AudioSegment

    print("✂️ Loading and splitting audio file...")
    audio = AudioSegment.from_file(file_path)

    chunk_duration = 10 * 60 * 1000  # 10 minutes
    chunks = [audio[i:i+chunk_duration] for i in range(0, len(audio), chunk_duration)]
    print(f"📦 Split into {len(chunks)} chunks")

    full_transcript = []

    for i, chunk in enumerate(chunks):
        print(f"🎙️ Transcribing chunk {i+1}/{len(chunks)}...")
        chunk_path = f"temp_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")

        # Retry logic for rate limits
        for attempt in range(3):
            try:
                with open(chunk_path, "rb") as f:
                    result = client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=f,
                        language="ar",
                        response_format="text"
                    )
                full_transcript.append(result)
                os.remove(chunk_path)
                break  # success, move to next chunk

            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = (attempt + 1) * 60  # wait 60s, 120s, 180s
                    print(f"⏳ Rate limit hit — waiting {wait} seconds...")
                    time.sleep(wait)
                else:
                    raise e  # different error, raise it

        # Small pause between chunks to avoid hitting limits
        if i < len(chunks) - 1:
            print("⏸️ Waiting 10 seconds before next chunk...")
            time.sleep(10)

    print("✅ All chunks transcribed!")
    return " ".join(full_transcript)