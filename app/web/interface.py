import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import shutil
from datetime import datetime
from app.ml.stt_engine import transcribe_audio
from app.ml.diarization_engine import run_diarization

st.set_page_config(page_title="Summarizer")

st.title("Summarizer")

UPLOAD_DIR = os.path.join("data", "uploads")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

uploaded_file = st.file_uploader("Выберите аудиофайл встречи для суммаризации (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{uploaded_file.name}")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Файл успешно сохранен в локальное хранилище: {file_path}")
    
    st.info("Файл готов к обработке")
    if st.button("Начать обработку"):
        try:
            with st.spinner("Определение голосов..."):
                diar_map = run_diarization(file_path)
        except Exception as e:
            st.error(f"Ошибка при определении голосов: {e}")
            
        try:
            with st.spinner("Расшифровка текста..."):
                segments = transcribe_audio(file_path)
        except Exception as e:
            st.error(f"Ошибка при расшифровке текста: {e}")    
            
        st.success("Обработка завершена!")
    
        final_transcript = ""
    
        for segment in segments:
            mid_point = (segment.start + segment.end) / 2
            speaker = "Неизвестный"
        
            for turn, _, spk in diar_map.itertracks(yield_label=True):
                if turn.start <= mid_point <= turn.end:
                    speaker = spk
                    break
        
            entry = f"**[{speaker}]**: {segment.text}\n\n"
            final_transcript += entry
            st.markdown(entry)

        output_file = os.path.join("data", "outputs", f"{timestamp}_transcript.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_transcript)