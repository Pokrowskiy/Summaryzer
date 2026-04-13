import streamlit as st
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.ml.stt_engine import transcribe_audio
from app.ml.diarization_engine import run_diarization
from app.ml.summarizer import MeetingSummarizer, parse_llm_result

st.set_page_config(page_title="Summarizer AI", layout="wide")

st.title("Meeting Summarizer")

UPLOAD_DIR = os.path.join("data", "uploads")
OUTPUT_DIR = os.path.join("data", "outputs")
for d in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

uploaded_file = st.file_uploader("Выберите аудиофайл (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{uploaded_file.name}")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Файл загружен")
    
    if st.button("Начать обработку"):
        start_total = time.perf_counter()
        
        step_start = time.perf_counter()
        try:
            with st.status("Определение голосов...", expanded=False) as status:
                diar_map = run_diarization(file_path)
                dt = time.perf_counter() - step_start
                status.update(label=f"Голоса определены ({dt:.1f} сек.)", state="complete")
        except Exception as e:
            st.error(f"Ошибка диаризации: {e}")
            st.stop()

        step_start = time.perf_counter()
        try:
            with st.status("Расшифровка текста...", expanded=False) as status:
                segments = transcribe_audio(file_path)
                dt = time.perf_counter() - step_start
                status.update(label=f"Текст расшифрован ({dt:.1f} сек.)", state="complete")
        except Exception as e:
            st.error(f"Ошибка транскрипции: {e}")
            st.stop()

        final_transcript = ""
        for segment in segments:
            mid_point = (segment.start + segment.end) / 2
            speaker = "Неизвестный"
            for turn, _, spk in diar_map.itertracks(yield_label=True):
                if turn.start <= mid_point <= turn.end:
                    speaker = spk
                    break
            final_transcript += f"**[{speaker}]**: {segment.text}\n\n"

        step_start = time.perf_counter()
        try:
            with st.status("Идёт анализ расшифровки...", expanded=False) as status:
                summarizer = MeetingSummarizer()
                raw_analysis = summarizer.process_transcript(final_transcript)
                analysis_data = parse_llm_result(raw_analysis)
                dt = time.perf_counter() - step_start
                status.update(label=f"Анализ завершен ({dt:.1f} сек.)", state="complete")
        except Exception as e:
            st.error(f"Ошибка LLM: {e}")
            st.stop()

        total_time = time.perf_counter() - start_total
        st.divider()

        st.subheader(f"Итоги встречи (Общее время обработки: {total_time:.1f} сек.)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("Участники встречи :")
            for tag, name in analysis_data["names"].items():
                st.write(f"**{tag}** — {name}")
            
            st.markdown("Задачи")
            for task in analysis_data["tasks"]:
                st.checkbox(task, key=task[:20] + str(time.time()))

        with col2:
            st.markdown("Саммари")
            st.info(analysis_data["summary"])
            
            with st.expander("Полный транскрипт"):
                st.markdown(final_transcript)

        output_prefix = os.path.join(OUTPUT_DIR, f"{timestamp}")
        
        with open(f"{output_prefix}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(final_transcript)
            
        with open(f"{output_prefix}_analysis.txt", "w", encoding="utf-8") as f:
            f.write(f"РЕЗЮМЕ:\n{analysis_data['summary']}\n\nЗАДАЧИ:\n" + "\n".join(analysis_data['tasks']))

        st.success(f"Результаты сохранены в папку {OUTPUT_DIR}")