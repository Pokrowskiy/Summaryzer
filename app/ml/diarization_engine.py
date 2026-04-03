import sys
import os
import torch
import torchaudio
import numpy as np
from dotenv import load_dotenv

# --- ОПЕРАЦИЯ ПО СПАСЕНИЮ ИМПОРТОВ ---

# 1. Исправляем структуру torchaudio в памяти
import types
if not hasattr(torchaudio, "backend"):
    # Создаем модуль-пустышку
    backend = types.ModuleType("torchaudio.backend")
    sys.modules["torchaudio.backend"] = backend
    
    # Создаем внутри него common
    common = types.ModuleType("torchaudio.backend.common")
    sys.modules["torchaudio.backend.common"] = common
    
    # Добавляем функции-заглушки в сам torchaudio
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
    torchaudio.get_audio_backend = lambda *args, **kwargs: "soundfile"
    torchaudio.list_audio_backends = lambda *args, **kwargs: ["soundfile"]

# 2. Исправляем NumPy 2.x
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# --- ТЕПЕРЬ ИМПОРТИРУЕМ ---
try:
    from pyannote.audio import Pipeline
except Exception as e:
    print(f"Критическая ошибка импорта: {e}")
    raise

load_dotenv()

def run_diarization(file_path):
    token = os.getenv("HF_TOKEN")
    
    # Пытаемся загрузить модель. В новых версиях чаще 'token'
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    print(f"--- Диаризация запущена на {device} ---")
    return pipeline(file_path)






def assign_speakers(whisper_results, diarization):
    """
    Сопоставляет сегменты Whisper с картой спикеров Pyannote
    """
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