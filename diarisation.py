import os
import tempfile
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper

TEMP_DIR = os.getenv("APP_TEMP_DIR", tempfile.gettempdir())
os.makedirs(TEMP_DIR, exist_ok=True)
tempfile.tempdir = TEMP_DIR

class SpeakerDiarizer:
    def __init__(self, auth_token, num_speakers=2, whisper_model_size="small"):
        self.num_speakers = num_speakers
        print("Loading diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
        print(f"Loading Whisper model ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)

    def diarize(self, audio_path_or_stream):
        print(f"Running diarization on: {audio_path_or_stream}")
        if isinstance(audio_path_or_stream, str):
            diarization = self.pipeline(audio_path_or_stream)
        else:
            diarization = self.pipeline({'uri': 'input', 'audio': audio_path_or_stream})
        speaker_map = self.assign_fixed_speaker_ids(diarization)
        transcript = self.build_transcript(audio_path_or_stream, diarization, speaker_map)
        return transcript

    def assign_fixed_speaker_ids(self, diarization):
        speaker_map = {}
        for turn, _, label in diarization.itertracks(yield_label=True):
            if label not in speaker_map:
                fixed_label = f"SPEAKER_{len(speaker_map)}"
                speaker_map[label] = fixed_label
                if len(speaker_map) >= self.num_speakers:
                    break
        return speaker_map

    def build_transcript(self, audio_path, diarization, speaker_map):
        audio = AudioSegment.from_file(audio_path)
        transcript = ""
        last_speaker = None

        for turn, _, label in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration < 0.5:
                continue  # Skip very short segments

            fixed_label = speaker_map.get(label, "UNKNOWN")
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            segment_audio = audio[start_ms:end_ms]
            tmp_path = tempfile.mktemp(suffix=".wav")

            try:
                segment_audio.export(tmp_path, format="wav")
                result = self.whisper_model.transcribe(tmp_path, language='en')
                segment_text = result["text"].strip()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if last_speaker != fixed_label:
                transcript += f"\n{fixed_label} [{self.format_time(turn.start)} - {self.format_time(turn.end)}]:\n"
                last_speaker = fixed_label

            transcript += segment_text + " "

        return transcript.strip()

    @staticmethod
    def format_time(seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
