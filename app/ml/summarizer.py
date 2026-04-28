import os
import time
import re
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()


REPO_ID = os.getenv("MODEL_REPO")             #"bartowski/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILENAME = os.getenv("MODEL_NAME")      #"Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MODELS_DIR = "models"

os.environ['GGML_PYTHON_LOG_LEVEL'] = 'ERROR'

class MeetingSummarizer:
    def __init__(self):
        print("Loading LLM on CPU...")
        self.model_path = self._ensure_model_exists()

        total_cores = os.cpu_count() or 4
        calculated_threads = max(total_cores - 4, int(total_cores * 0.75))
        # Модель использует на 4 ядра меньше доступных или не более 75% системы(в зависимости от того, чего больше), чтобы не занимать все мощности пользователя.
        self.n_threads = max(1, calculated_threads)
        print(f"System cores: {total_cores}, allocated threads: {self.n_threads}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=self.n_threads,
            verbose=False
        )
        print("LLM module ready.")

    def _ensure_model_exists(self):
        if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
        full_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        if not os.path.exists(full_path):
            print("LLM weights nor found. \n Dowlnloading weights...")
            hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, local_dir=MODELS_DIR)
        return full_path

    def process_transcript(self, text):
        prompt = f"""
<|im_start|>system
Ты - модуль автоматической обработки текста. Твоя задача: прочитать транскрипт и заполнить форму.
НЕ пиши вступлений. НЕ пиши заключений. Используй только данные из текста.
ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ.

ФОРМАТ:
[NAMES]
[SPEAKER_XX] - Имя (если упоминалось, иначе попробуй предположить его роль или укажи "Неизвестно")

[SUMMARY]
Краткое описание сути встречи на русском.

[TASKS]
- Задача 1
- Задача 2
<|im_end|>
<|im_start|>user
Транскрипт:
{text}
<|im_end|>
<|im_start|>assistant
"""
        response = self.llm(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["<|im_end|>"]
        )
        return response['choices'][0]['text']

def parse_llm_result(raw_text):
    result = {
        "names": {},
        "summary": "Резюме не сформировано",
        "tasks": []
    }

    names_block = re.search(r'\[NAMES\](.*?)(?:\[SUMMARY\]|РЕЗЮМЕ:)', raw_text, re.DOTALL | re.IGNORECASE)
    if names_block:
        lines = names_block.group(1).strip().split('\n')
        for line in lines:
            if '—' in line or ':' in line or '-' in line:
                match = re.search(r'SPEAKER_(\d+)', line, re.IGNORECASE)
                if match:
                    name = re.split(r'[—:-]', line)[-1].strip()
                    result["names"][f"SPEAKER_{match.group(1)}"] = name

    summary_block = re.search(r'(?:\[SUMMARY\]|РЕЗЮМЕ:)(.*?)(?:\[TASKS\]|ЗАДАЧИ:)', raw_text, re.DOTALL | re.IGNORECASE)
    if summary_block:
        result["summary"] = summary_block.group(1).strip()

    tasks_block = re.search(r'(?:\[TASKS\]|ЗАДАЧИ:)(.*)', raw_text, re.DOTALL | re.IGNORECASE)
    if tasks_block:
        lines = tasks_block.group(1).strip().split('\n')
        result["tasks"] = [l.strip('-•* ').strip() for l in lines if l.strip()]

    return result