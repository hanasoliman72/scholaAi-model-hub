import sys
import os
from transcriber import transcribe_audio, compress_audio
from summarizer import summarize_session


def save_results(transcript: str, summary: str, output_dir: str = "output"):
    """Save transcript and summary to text files."""
    os.makedirs(output_dir, exist_ok=True)

    transcript_path = os.path.join(output_dir, "transcript.txt")
    summary_path = os.path.join(output_dir, "summary.txt")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n💾 Transcript saved to: {transcript_path}")
    print(f"💾 Summary saved to:    {summary_path}")


def main():
    # --- Get file path from user ---
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("📂 Enter the path to your audio/video file: ").strip()

    # Validate file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    print("\n" + "="*50)
    print("  Arabic Session Summarizer")
    print("="*50 + "\n")

    # Step 1: Transcribe
    compressed_path = compress_audio(file_path)
    transcript = transcribe_audio(compressed_path)

    # Clean up compressed file after done
    if os.path.exists("compressed_audio.mp3"):
        os.remove("compressed_audio.mp3")

    print(f"\n📄 Transcript Preview (first 300 chars):")
    print("-" * 40)
    print(transcript[:300] + "..." if len(transcript) > 300 else transcript)

    # Step 2: Summarize
    summary = summarize_session(transcript)

    print(f"\n📋 FULL SUMMARY REPORT:")
    print("=" * 50)
    print(summary)

    # Step 3: Save to files
    save_results(transcript, summary)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()