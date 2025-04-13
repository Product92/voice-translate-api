from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Render!"

@app.route('/upload', methods=['POST'])
def upload_audio():
    data = request.json
    audio_url = data.get("audio_url", "")
    # Ici tu feras ton traitement (téléchargement audio, mixage, etc.)
    return jsonify({"success": True, "received_audio_url": audio_url})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
