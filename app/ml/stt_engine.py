from faster_whisper import WhisperModel
import os

MODEL_SIZE = "base" 

def transcribe_audio(file_path):
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    segments, info = model.transcribe(file_path, beam_size=5)
    return list(segments)