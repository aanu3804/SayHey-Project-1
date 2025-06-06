from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import requests
from diarisation import SpeakerDiarizer
from drive_upload import upload_file_to_drive, make_file_public,delete_file_from_drive
from dotenv import load_dotenv
import tempfile
import io

load_dotenv()

# --- CONFIGURATION ---

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
with open("PERFECT_SESSIONS.txt", "r", encoding="utf-8") as f:
    PERFECT_SESSIONS = f.read()

app = Flask(__name__)

# Audio settings
ALLOWED_EXTENSIONS = {'wav'}
diarizer = SpeakerDiarizer(auth_token=HF_AUTH_TOKEN, num_speakers=2)

# --- HELPER FUNCTIONS ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(public_url):
    response = requests.get(public_url)
    if response.status_code != 200:
        raise Exception("Failed to download audio from Drive")
    audio_stream = io.BytesIO(response.content)
    transcript = diarizer.diarize(audio_stream)
    print(transcript)
    return transcript

def analyze_sentiment(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a summarizer. Strictly return only the summary of this conversation. "
                    "Do not add any introductory phrases like 'Here is the summary' or bullet points or bold points. "
                    "Label the person who is telling their problem as a USER and the other person as Listener."
                    "Dont give brief summary. Cover all discussed points, covering key matters.\n\n"
                    f"Transcript:\n{text}"
                )
            }
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print("Groq API error:", response.text)
        return "Error: Unable to analyze sentiment."

def generate_recommendation(summary):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conversation coach for empathetic support. Based on this conversation summary, give recommendations to the LISTENER. "
                    "Guide the LISTENER on how they can improve their tone, validation, reframing, and emotional support. Do not speak to the user.\n\n"
                    "Give the recommendations in a friendly and understanding tone. "
                    "Do not use phrases like 'you should', 'you can', 'remember', or 'it's okay'. "
                    "Avoid speaking as if responding to the user. Simply provide the best possible recommendations very briefly in less than 60 words.\n\n"
                )
            },
            {
                "role": "user",
                "content": f"Summary of conversation:\n{summary}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print("Groq API error:", response.text)
        return "Error: Unable to generate recommendation."

def rate_listener(summary):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        "You are a rating assistant. Rate the LISTENER in the following mental health support categories out of 5 based on the conversation summary. "
        "Use the perfect sessions as a reference standard.\n"
        "Categories: Opening Warmth, Active Listening, Validation, Gentle Probing, Reframing, Emotion Check-in, Closure. "
        "Give output with field names and numerical values from 1 to 5.\n\n"
        "Don't specify any introductory phrases like 'Here is the rating for Listener:', just give the Category name and rating number between 1-5 only .\n\n"
        f"Perfect Examples:\n{PERFECT_SESSIONS}\n\nSummary:\n{summary}"
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print("Groq API error:", response.text)
        return "{}"

# --- ROUTES ---

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    past_summary = request.form.get('past_summary', '').strip()

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)

        # Upload to Google Drive and make public
        FOLDER_ID = os.getenv("FOLDER_ID")
        drive_id = upload_file_to_drive(temp_path, filename, folder_id=FOLDER_ID)
        public_url = make_file_public(drive_id)
# Download the uploaded file from Drive and reprocess from it
        transcript = transcribe_audio(public_url)
        sentiment_analysis = analyze_sentiment(transcript)
        recommendation = generate_recommendation(sentiment_analysis)
        rating = rate_listener(sentiment_analysis)

        # Cleanup
        delete_file_from_drive(drive_id)



        return jsonify({
            'public_audio_url': public_url,
            'transcript': transcript,
            'sentiment_analysis': sentiment_analysis,
            'recommendation': recommendation,
            'rating': rating
        })

    else:
        return jsonify({'error': 'Invalid file format. Please upload a WAV file.'})

# --- MAIN ---
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=PORT)
