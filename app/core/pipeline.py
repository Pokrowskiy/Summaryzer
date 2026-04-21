import os
from app.ml.diarization import DiarizationEngine
from app.ml.transcription import WhisperEngine
from app.ml.summarizer import MeetingSummarizer, parse_llm_result

class MeetingAnalysisPipeline:
    def __init__(self):
        self.diarizer = DiarizationEngine()
        self.transcriber = WhisperEngine()
        self.summarizer = MeetingSummarizer()

    def process_meeting(self, audio_path):
        print("--- Шаг 1: Анализ голосов (Диаризация) ---")
        diarization_segments = self.diarizer.process(audio_path)
        
        print("--- Шаг 2: Преобразование речи в текст ---")
        full_transcript = self.transcriber.transcribe_with_speakers(audio_path, diarization_segments)
        
        print("--- Шаг 3: Генерация резюме и поиск имен ---")
        raw_analysis = self.summarizer.summarize(full_transcript)
        
        final_data = parse_llm_result(raw_analysis)
        
        return {
            "transcript": full_transcript,
            "analysis": final_data
        }