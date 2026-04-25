from fastapi import FastAPI, BackgroundTasks
import time
import threading
import os
import json

app = FastAPI()

from app.ml.stt_engine import WhisperEngine
from app.ml.diarization_engine import DiarizationEngine
from app.ml.summarizer import MeetingSummarizer, parse_llm_result

status = {"models_loaded": False}
tasks_db = {}

MODELS = {
    "diarizer": None,
    "stt": None,
    "llm": None
}

def load_models():
    MODELS["diarizer"] = DiarizationEngine() 
    MODELS["stt"] = WhisperEngine()
    MODELS["llm"] = MeetingSummarizer()
    
    status["models_loaded"] = True
    print("Backend: Models loaded.")

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_models, daemon=True).start()



def run_pipeline(file_name: str):
    try:
        os.makedirs("data/outputs", exist_ok=True)
        os.makedirs("data/db", exist_ok=True)
        input_path = f"data/uploads/{file_name}"
        output_path = f"data/outputs/{file_name}.json"

        tasks_db[file_name] = "diarization"
        diar_result = MODELS["diarizer"].process(input_path)

        tasks_db[file_name] = "transcription"
        whisper_segments = MODELS["stt"].transcribe(input_path)
        
        full_transcript = MODELS["diarizer"].assign_speakers(whisper_segments, diar_result)

        tasks_db[file_name] = "summarization"
        
        text = "\n".join([
            f"[{s['speaker']}]: {s['text']}" for s in full_transcript
        ])
        with open(f"data/outputs/{file_name}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(text)
        
        raw_output = MODELS["llm"].process_transcript(text)
        output = parse_llm_result(raw_output)
        with open(f"data/outputs/{file_name}_analysys.txt", "w", encoding="utf-8") as f:
            f.write(raw_output)

        output["full_transcript"] = full_transcript
        with open(f"data/db/{file_name}.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        tasks_db[file_name] = "completed"

    except Exception as e:
        print(f"Ошибка при обработке {file_name}: {e}")
        tasks_db[file_name] = f"error: {str(e)}"





@app.get("/health")
async def health_check():
    return {"ready": status["models_loaded"]}

@app.post("/process")
async def process_file(file_name: str, background_tasks: BackgroundTasks):
    if not status["models_loaded"]:
        return {"error": "Models not ready"}
    tasks_db[file_name] = "pending"
    background_tasks.add_task(run_pipeline, file_name)
    return {"message": "Started"}

@app.get("/status/{file_name}")
async def get_task_status(file_name: str):
    return {"status": tasks_db.get(file_name, "idle")}

@app.get("/result/{file_name}")
async def get_result(file_name: str):
    path = f"data/db/{file_name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Not found"}