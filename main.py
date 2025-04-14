from flask import Flask, request, jsonify
import os
import subprocess
import assemblyai

app = Flask(__name__)

# Set AssemblyAI API key from environment variable
assemblyai.api_key = os.getenv("ASSEMBLYAI_API_KEY")

def download_audio(video_url, output_path="temp_audio.mp3"):
    """Download audio from a video URL using yt-dlp"""
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "-o", output_path,
        video_url
    ]
    subprocess.run(cmd, check=True)
    return output_path

def transcribe_audio(file_path):
    """Transcribe audio using AssemblyAI"""
    transcriber = assemblyai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    return transcript.json()

@app.route("/")
def home():
    return "Translation API is running!"

@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        video_url = data.get("video_url", "")

        if not video_url:
            return jsonify({"success": False, "error": "Missing video_url"}), 400

        # Step 1: Download audio
        print("Downloading audio...")
        audio_path = download_audio(video_url)

        # Step 2: Transcribe with AssemblyAI
        print("Transcribing audio...")
        transcription = transcribe_audio(audio_path)

        # Optional: Clean up temp audio file
        os.remove(audio_path)

        # Step 3: Return result
        return jsonify({
            "success": True,
            "video_url": video_url,
            "transcription": transcription  # Contains text, speakers, timestamps, etc.
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
