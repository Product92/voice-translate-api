from flask import Flask, request, jsonify
import os
import subprocess
import json
import time
import assemblyai
import requests
import uuid
import tempfile
from pydub import AudioSegment
import openai
import boto3
from botocore.client import Config

app = Flask(__name__)

# Environment variables (set these in your Render dashboard)
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
CLOUDFLARE_ENDPOINT = os.getenv("CLOUDFLARE_ENDPOINT")
CLOUDFLARE_ACCESS_KEY_ID = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
CLOUDFLARE_SECRET_ACCESS_KEY = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY")
CLOUDFLARE_BUCKET_NAME = os.getenv("CLOUDFLARE_BUCKET_NAME")

# Initialize clients
assemblyai.api_key = ASSEMBLYAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Initialize Cloudflare R2 client (S3 compatible)
r2 = boto3.client(
    's3',
    endpoint_url=CLOUDFLARE_ENDPOINT,
    aws_access_key_id=CLOUDFLARE_ACCESS_KEY_ID,
    aws_secret_access_key=CLOUDFLARE_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

def download_audio(video_url, output_path):
    """Download audio from a video URL using yt-dlp"""
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "5",  # Lower quality for faster processing
        "-o", output_path,
        video_url
    ]
    subprocess.run(cmd, check=True)
    return output_path

def transcribe_audio(file_path):
    """Transcribe audio using AssemblyAI with speaker detection"""
    transcriber = assemblyai.Transcriber()
    config = assemblyai.TranscriptionConfig(
        speaker_labels=True,  # Enable speaker detection
        audio_start_from=0,
        audio_end_at=None,
    )
    
    transcript = transcriber.transcribe(file_path, config=config)
    
    # Wait for transcription to complete
    while transcript.status != "completed":
        time.sleep(5)
        transcript = transcriber.get_transcript(transcript.id)
        
    return transcript.json()

def translate_transcript(transcript_data, target_language="fr"):
    """Translate transcript using OpenAI's GPT"""
    translated_utterances = []
    
    # Extract utterances with speaker info
    utterances = transcript_data.get("utterances", [])
    
    for utterance in utterances:
        speaker = utterance.get("speaker", "unknown")
        text = utterance.get("text", "")
        start = utterance.get("start", 0)
        end = utterance.get("end", 0)
        
        # Translate text using OpenAI
        prompt = f"""Please translate the following text to {target_language}, 
                   maintaining the original tone, emotions and context:
                   "{text}"
                   """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        
        translated_text = response.choices[0].message.content.strip('"')
        
        translated_utterances.append({
            "speaker": speaker,
            "original_text": text,
            "translated_text": translated_text,
            "start": start,
            "end": end
        })
    
    return translated_utterances

def generate_voice(text, speaker_type, voice_id=None):
    """Generate voice using ElevenLabs API"""
    # Default voices based on speaker type
    voice_mapping = {
        "A": "pNInz6obpgDQGcFmaJgB",  # Adam for male
        "B": "EXAVITQu4vr4xnSDxMaL",  # Sarah for female
        "C": "IKne3meq5aSn9XLyUdCD",  # Josh for child/other
    }
    
    # Use provided voice_id or default based on speaker_type
    selected_voice = voice_id if voice_id else voice_mapping.get(speaker_type, voice_mapping["A"])
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        # Create temporary file to store audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    else:
        raise Exception(f"ElevenLabs API error: {response.text}")

def mix_audio(original_audio_path, voice_clips, output_path):
    """Mix original audio with AI-generated voice clips"""
    # Load the original audio
    original_audio = AudioSegment.from_file(original_audio_path)
    
    # Reduce volume of original audio by 70%
    original_audio = original_audio - 14  # Approximately -70% in dB
    
    # Create a new audio segment with the original audio
    mixed_audio = original_audio
    
    # Overlay each voice clip at its respective timestamp
    for clip in voice_clips:
        voice_audio = AudioSegment.from_file(clip["file_path"])
        position_ms = clip["start"]
        mixed_audio = mixed_audio.overlay(voice_audio, position=position_ms)
    
    # Export the mixed audio
    mixed_audio.export(output_path, format="mp3")
    return output_path

def upload_to_cloudflare(file_path):
    """Upload file to Cloudflare R2 storage"""
    file_name = f"translated_{uuid.uuid4().hex}.mp3"
    
    with open(file_path, 'rb') as file_data:
        r2.upload_fileobj(
            file_data, 
            CLOUDFLARE_BUCKET_NAME, 
            file_name,
            ExtraArgs={'ContentType': 'audio/mp3', 'ACL': 'public-read'}
        )
    
    # Generate public URL for the uploaded file
    file_url = f"{CLOUDFLARE_ENDPOINT}/{CLOUDFLARE_BUCKET_NAME}/{file_name}"
    return file_url

def cleanup_temp_files(file_list):
    """Remove temporary files"""
    for file_path in file_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

@app.route("/")
def home():
    return "IA Translation Pipeline API is running!"

@app.route("/translate", methods=["POST"])
def translate_video():
    try:
        data = request.get_json()
        video_url = data.get("video_url", "")
        target_language = data.get("target_language", "fr")
        custom_voices = data.get("custom_voices", {})  # Optional voice IDs mapping
        
        if not video_url:
            return jsonify({"success": False, "error": "Missing video_url"}), 400
        
        temp_files = []  # To track files for cleanup
        
        # Step 1-2: Download audio from video
        print("Downloading audio...")
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{uuid.uuid4().hex}.mp3")
        audio_path = download_audio(video_url, temp_audio_path)
        temp_files.append(audio_path)
        
        # Step 3: Transcribe with AssemblyAI
        print("Transcribing audio...")
        transcription = transcribe_audio(audio_path)
        
        # Step 4: Translate transcript using GPT
        print("Translating content...")
        translated_utterances = translate_transcript(transcription, target_language)
        
        # Step 5: Generate AI voices for each utterance
        print("Generating voices...")
        voice_clips = []
        
        for utterance in translated_utterances:
            speaker = utterance["speaker"]
            text = utterance["translated_text"]
            start = utterance["start"]
            
            # Use custom voice if provided, otherwise use default mapping
            custom_voice_id = custom_voices.get(speaker)
            
            voice_file = generate_voice(text, speaker, custom_voice_id)
            temp_files.append(voice_file)
            
            voice_clips.append({
                "file_path": voice_file,
                "start": start,
                "speaker": speaker
            })
        
        # Step 6: Mix the original audio with AI voices
        print("Mixing audio...")
        output_path = os.path.join(tempfile.gettempdir(), f"final_mix_{uuid.uuid4().hex}.mp3")
        mixed_audio_path = mix_audio(audio_path, voice_clips, output_path)
        temp_files.append(mixed_audio_path)
        
        # Step 7: Upload to Cloudflare R2
        print("Uploading to Cloudflare R2...")
        file_url = upload_to_cloudflare(mixed_audio_path)
        
        # Step 8: Return the final result
        result = {
            "success": True,
            "video_url": video_url,
            "file_url": file_url,
            "original_transcription": transcription,
            "translated_content": translated_utterances,
            "player_html": f'<audio controls src="{file_url}"></audio>'
        }
        
        # Clean up temporary files
        cleanup_temp_files(temp_files)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
