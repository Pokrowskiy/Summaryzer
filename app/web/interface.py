import streamlit as st
import requests
import time
import os
from datetime import datetime

st.set_page_config(page_title="Summarizer", layout="wide")
API_URL = "http://127.0.0.1:8000"

def check_backend():
    try:
        return requests.get(f"{API_URL}/health").json()
    except: return None

backend_state = check_backend()
if not backend_state or not backend_state.get("ready"):
    st.warning("Идёт инициализация весов на бэкенде...")
    time.sleep(2)
    st.rerun()

st.title("Summarizer")
uploaded_file = st.file_uploader("Загрузите аудио для обработки", type=["mp3", "wav", "m4a"])

if uploaded_file:
    os.makedirs("data/uploads", exist_ok=True)
    timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    timed_filename = f"{timestamp}_{uploaded_file.name}"
    with open(f"data/uploads/{timed_filename}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Запустить обработку"):
        requests.post(f"{API_URL}/process", params={"file_name": timed_filename})
        
        status_box = st.empty()
        bar = st.progress(0)

        timer_text = st.empty()
        start_process_time = time.time()
        
        while True:
            res = requests.get(f"{API_URL}/status/{timed_filename}").json()
            state = res.get("status")

            elapsed_time = int(time.time() - start_process_time)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            timer_text.caption(f"Идёт обработка : {minutes:02d}:{seconds:02d}")

            if state == "diarization":
                status_box.info("Идёт распознавание голосов...")
                bar.progress(10)
            elif state == "transcription":
                status_box.info("Идёт расшифровка аудио в текст...")
                bar.progress(40)
            elif state == "summarization":
                status_box.info("Идёт анализ расшифровки...")
                bar.progress(50)
            elif state == "completed":
                status_box.success("Обработка завершена!")
                bar.progress(100)
                result = requests.get(f"{API_URL}/result/{timed_filename}").json()
                
                st.divider()
                status_box.success(f"Общее время обработки: {minutes:02d}:{seconds:02d}")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Участники")
                    for spk, name in result["names"].items():
                        st.write(f"**{spk}**: {name}")
                    st.subheader("Задачи")
                    for task in result["tasks"]:
                        st.write(f"- {task}")
                with col2:
                    st.subheader("Резюме")
                    st.write(result["summary"])
                st.divider()
                st.success("Результаты сохранены в папку data/outputs/")
                break

            elif "error" in state:
                st.error(f"Ошибка: {state}")
                break
            time.sleep(1)