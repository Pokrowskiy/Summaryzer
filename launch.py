import subprocess
import sys
import time
import signal
import os

def start_services():
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "app.api.main:app", 
        "--host", "127.0.0.1", "--port", "8000"
    ]
    
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", "app/web/interface.py", 
        "--server.port", "8501", "--browser.gatherUsageStats", "false"
    ]

    print("Запуск сервисов...")
    
    backend_proc = subprocess.Popen(backend_cmd)
    time.sleep(2)
    frontend_proc = subprocess.Popen(frontend_cmd)

    print("Система работает. Нажмите Ctrl+C для выхода.")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None or frontend_proc.poll() is not None:
                break
    except KeyboardInterrupt:
        print("Получен сигнал остановки.")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        os.system(f"taskkill /F /PID {backend_proc.pid} /T >nul 2>&1")
        os.system(f"taskkill /F /PID {frontend_proc.pid} /T >nul 2>&1")

if __name__ == "__main__":
    start_services()