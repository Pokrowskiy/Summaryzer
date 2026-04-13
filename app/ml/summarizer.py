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
            n_threads=12, 
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
            
        prompt = prompt = f"""<|im_start|>system
            Ты — профессиональный аналитик встреч. Твоя задача — извлечь данные из транскрипта в строго заданном формате, данные нужны на русском языке.

            ИНСТРУКЦИИ:
            1. Сопоставь [SPEAKER_XX] с реальными именами на основе контекста.
            2. Напиши краткое резюме (3-5 предложений).
            3. Составь список конкретных поручений.

            ВЫВОДИ ОТВЕТ СТРОГО ПО ШАБЛОНУ:

            [NAMES]
            [SPEAKER_00] — Имя
            [SPEAKER_01] — Имя

            [SUMMARY]
            Текст резюме встречи.

            [TASKS]
            - Имя испольнителя задачи: Задача 1
            - Имя испольнителя задачи: Задача 2

            Никаких вступительных фраз. Только блоки данных.<|im_end|>
            <|im_start|>user
            Транскрипт:
            {text}<|im_end|>
            <|im_start|>assistant
        """
        start_time = time.perf_counter()
        response = self.llm(
            f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
            max_tokens=1024,
            temperature=0.2,
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
    """
    Парсит структурированный ответ от LLM.
    Устойчив к мелким опечаткам в тегах [SPEAKER_XX].
    """
    result = {
        "names": {},
        "summary": "Резюме не сформировано",
        "tasks": []
    }
    
    # Разбиваем текст на блоки по заголовкам в скобках
    parts = re.split(r'\[(NAMES|SUMMARY|TASKS)\]', raw_text, flags=re.IGNORECASE)
    
    current_section = None
    for i in range(len(parts)):
        section_content = parts[i].strip()
        
        if section_content.upper() == "NAMES":
            current_section = "NAMES"
        elif section_content.upper() == "SUMMARY":
            current_section = "SUMMARY"
        elif section_content.upper() == "TASKS":
            current_section = "TASKS"
        else:
            # Обработка содержимого секций
            if current_section == "NAMES":
                # Ищем строки вида [SPEAKER_00] — Имя или SPEAKER_00: Имя
                lines = section_content.split('\n')
                for line in lines:
                    if '—' in line or ':' in line or '-' in line:
                        # Улучшенное регулярное выражение для поиска SPEAKER_XX
                        match = re.search(r'SPE\w+?_(\d+)', line, re.IGNORECASE)
                        if match:
                            spk_id = f"SPEAKER_{match.group(1)}"
                            # Берем всё, что после разделителя
                            name = re.split(r'[—:-]', line)[-1].strip()
                            result["names"][spk_id] = name

            elif current_section == "SUMMARY":
                if section_content:
                    result["summary"] = section_content

            elif current_section == "TASKS":
                # Ищем строки, начинающиеся с буллитов
                lines = section_content.split('\n')
                for line in lines:
                    clean_line = line.strip().lstrip('-•*').strip()
                    if clean_line:
                        result["tasks"].append(clean_line)
                        
    return result