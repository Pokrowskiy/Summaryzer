import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from app.ml.summarizer import MeetingSummarizer

def run_test():
    transcript_path = current_file.parent / "testing_transcript.txt"
    
    if not transcript_path.exists():
        print(f"Ошибка: Файл '{transcript_path}' не найден.")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        summarizer = MeetingSummarizer()
        result = summarizer.process_transcript(content)

        print("="*40)
        print(result)
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Ошибка в процессе теста: {e}")

if __name__ == "__main__":
    run_test()