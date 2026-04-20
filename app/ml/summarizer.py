import os
import time
import re
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

REPO_ID = "bartowski/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILENAME = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MODELS_DIR = "models"

os.environ['GGML_PYTHON_LOG_LEVEL'] = 'ERROR'

class MeetingSummarizer:
    def __init__(self):
        self.model_path = self._ensure_model_exists()
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=16, 
            verbose=True
        )

    def _ensure_model_exists(self):
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            
        full_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        
        if not os.path.exists(full_path):
            print(f"Модель не найдена. Идёт загрузка модели...")
            print("Это займет некоторое время и займёт около 2 ГБ на диске, но необходимо только один раз.")
            
            hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILENAME,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False
            )
            print("Модель загружена и готова к работе.")
            
        return full_path

    def process_transcript(self, text):
        llm = Llama(model_path=self.model_path, n_ctx=2048, n_threads=4) 
            
        prompt = f"""<|im_start|>system
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
        start_time = time.perf_counter()
        response = self.llm(
            f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
            max_tokens=1024,
            temperature=0.3,
            stop=["<|end|>"],
            echo=False
        )
        end_time = time.perf_counter()
        tokens_count = response['usage']['completion_tokens']
        duration = end_time - start_time
        
        tokens_per_sec = tokens_count / duration if duration > 0 else 0
        
        print(f"Сгенерировано токенов: {tokens_count}")
        print(f"Время генерации: {duration:.2f} сек.")
        print(f"Скорость: {tokens_per_sec:.2f} токенов/сек.")

        return response['choices'][0]['text']

def parse_llm_result(raw_text):
    result = {
        "names": {},
        "summary": "Резюме не сформировано",
        "tasks": []
    }

    # Поиск имен (между NAMES и SUMMARY/РЕЗЮМЕ)
    names_block = re.search(r'\[NAMES\](.*?)(?:\[SUMMARY\]|РЕЗЮМЕ:)', raw_text, re.DOTALL | re.IGNORECASE)
    if names_block:
        lines = names_block.group(1).strip().split('\n')
        for line in lines:
            if '—' in line or ':' in line or '-' in line:
                match = re.search(r'SPEAKER_(\d+)', line, re.IGNORECASE)
                if match:
                    name = re.split(r'[—:-]', line)[-1].strip()
                    result["names"][f"SPEAKER_{match.group(1)}"] = name

    # Поиск саммари (между SUMMARY/РЕЗЮМЕ и TASKS/ЗАДАЧИ)
    summary_block = re.search(r'(?:\[SUMMARY\]|РЕЗЮМЕ:)(.*?)(?:\[TASKS\]|ЗАДАЧИ:)', raw_text, re.DOTALL | re.IGNORECASE)
    if summary_block:
        result["summary"] = summary_block.group(1).strip()

    # Поиск задач (всё после TASKS/ЗАДАЧИ)
    tasks_block = re.search(r'(?:\[TASKS\]|ЗАДАЧИ:)(.*)', raw_text, re.DOTALL | re.IGNORECASE)
    if tasks_block:
        lines = tasks_block.group(1).strip().split('\n')
        result["tasks"] = [l.strip('-•* ').strip() for l in lines if l.strip()]

    return result