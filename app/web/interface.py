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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"data/uploads/{timestamp}_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Запустить обработку"):
        requests.post(f"{API_URL}/process", params={"file_name": uploaded_file.name})
        
        status_box = st.empty()
        bar = st.progress(0)
        
        while True:
            res = requests.get(f"{API_URL}/status/{uploaded_file.name}").json()
            state = res.get("status")

            if state == "diarization":
                status_box.info("Идёт распознавание голосов...")
                bar.progress(20)
            elif state == "transcription":
                status_box.info("Идёт расшифровка аудио в текст...")
                bar.progress(40)
            elif state == "summarization":
                status_box.info("Идёт анализ расшифровки...")
                bar.progress(60)
            elif state == "completed":
                status_box.success("Обработка завершена!")
                bar.progress(100)
                
                result = requests.get(f"{API_URL}/result/{uploaded_file.name}").json()
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Резюме")
                    st.write(result["summary"])
                with col2:
                    st.subheader("Участники")
                    for spk, name in result["names"].items():
                        st.write(f"**{spk}**: {name}")
                break
            elif "error" in state:
                st.error(f"Ошибка: {state}")
                break
            
            time.sleep(1)