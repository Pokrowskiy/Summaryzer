import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

# --- Блок 1: Юнит-тесты функционала ---

def test_health_check():
    """1. Проверка доступности бэкенда"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "ready" in response.json()

def test_core_count_logic():
    """2. Проверка логики распределения ядер"""
    total_cores = 16
    calculated_threads = max(total_cores - 4, int(total_cores * 0.75))
    assert calculated_threads == 12
    
    total_cores = 4
    calculated_threads = max(total_cores - 4, int(total_cores * 0.75))
    assert calculated_threads == 3

def test_env_loading():
    """3. Проверка загрузки переменных окружения"""
    assert os.getenv("MODEL_NAME") is not None
    assert "GGUF" in os.getenv("MODEL_REPO")

def test_directory_structure():
    """4. Проверка создания необходимых папок при старте"""
    required_dirs = ["data/uploads", "data/outputs", "data/db"]
    for d in required_dirs:
        assert os.path.exists(d)

def test_file_naming_convention():
    """5. Проверка формата именования файлов"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    assert len(timestamp.split("_")) == 2
    assert "." in timestamp


# --- Блок 2: Проверка невалидных данных (Edge Cases) ---

def test_process_non_existent_file():
    """Проверка запроса на обработку файла, которого нет в папке uploads"""
    response = client.post("/process", params={"file_name": "ghost_file.mp3"})
    assert response.status_code == 404

def test_invalid_status_request():
    """Запрос статуса для несуществующей задачи"""
    response = client.get("/status/fake_task_123")
    assert response.status_code == 404