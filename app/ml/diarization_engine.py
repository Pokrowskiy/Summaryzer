import sys
import os
import torch
import torchaudio
import numpy as np
from dotenv import load_dotenv

import types
if not hasattr(torchaudio, "backend"):
    backend = types.ModuleType("torchaudio.backend")
    sys.modules["torchaudio.backend"] = backend
    
    common = types.ModuleType("torchaudio.backend.common")
    sys.modules["torchaudio.backend.common"] = common
    
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
    torchaudio.get_audio_backend = lambda *args, **kwargs: "soundfile"
    torchaudio.list_audio_backends = lambda *args, **kwargs: ["soundfile"]

if not hasattr(np, "NaN"):
    np.NaN = np.nan

try:
    from pyannote.audio import Pipeline
except Exception as e:
    print(f"Критическая ошибка импорта: {e}")
    raise

from pyannote.audio import Pipeline

load_dotenv()

class DiarizationEngine:
    def __init__(self):
        token = os.getenv("HF_TOKEN")
        print("Loading Pyannote Pipeline...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token
            )
        except TypeError:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)
        print(f"Pyannote ready on {self.device}.")

    def process(self, file_path):
        return self.pipeline(file_path)

    @staticmethod
    def assign_speakers(whisper_results, diarization):
        final_segments = []
        for segment in whisper_results:
            mid_time = (segment.start + segment.end) / 2
            speaker = "Unknown"
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if turn.start <= mid_time <= turn.end:
                    speaker = spk
                    break
            
            final_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
                "text": segment.text
            })
        return final_segments