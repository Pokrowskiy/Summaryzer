import torch
from faster_whisper import WhisperModel

class WhisperEngine:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        print("Loading Whisper Model...")
        self.model = WhisperModel(
            model_size, 
            device=self.device, 
            compute_type=self.compute_type
        )
        print(f"Whisper model({model_size}) ready on {self.device}.")

    def transcribe(self, file_path):
        segments, info = self.model.transcribe(file_path, beam_size=5)
        return list(segments)