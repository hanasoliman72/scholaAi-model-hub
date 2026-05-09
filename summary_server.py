from fastapi import FastAPI
from pydantic import BaseModel
from transcriber import transcribe_audio, compress_audio
from summarizer import summarize_session
import requests
import os
import tempfile

# uvicorn summary_server:app --port 8001

app = FastAPI()

class SummaryRequest(BaseModel):
    video_url: str
    session_id: int

@app.post("/summarize")
async def summarize(req: SummaryRequest):
    try:
        # 1. Download audio from Supabase URL
        print(f"⬇️ Downloading from {req.video_url}")
        response = requests.get(req.video_url, stream=True)
        
        # save to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".webm", delete=False
        ) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        # 2. Compress audio
        compressed_path = compress_audio(tmp_path)

        # 3. Transcribe
        print("🎙️ Transcribing...")
        transcript = transcribe_audio(compressed_path)

        # 4. Summarize
        print("📝 Summarizing...")
        summary = summarize_session(transcript)

        # 5. Cleanup
        os.remove(tmp_path)
        os.remove(compressed_path)

        return {
            "success": True,
            "summary": summary,
            "transcript": transcript
        }

    except Exception as e:
        return {"success": False, "error": str(e)}