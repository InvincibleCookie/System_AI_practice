import os
import time
import json
import hashlib
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


# ============================
# History / Reports
# ============================
HISTORY_PATH = os.path.join(os.getcwd(), "history.json")


def _safe_read_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_history() -> list:
    return _safe_read_json(HISTORY_PATH)


def append_history(record: Dict[str, Any]) -> None:
    hist = load_history()
    hist.append(record)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def generate_excel_bytes(history: list) -> bytes:
    df = pd.json_normalize(history)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.close()
    with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="history")
    with open(tmp.name, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp.name)
    except Exception:
        pass
    return data


# ============================
# Utils: IOU + simple tracker
# ============================
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


@dataclass
class Track:
    tid: int
    box: np.ndarray  # xyxy
    last_seen: int
    missed: int = 0


class SimpleIOUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 20):
        self.iou_thresh = float(iou_thresh)
        self.max_missed = int(max_missed)
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[np.ndarray], frame_idx: int) -> List[Track]:
        unmatched_det = set(range(len(detections)))

        self.tracks.sort(key=lambda t: t.last_seen, reverse=True)
        for tr in self.tracks:
            best_iou = 0.0
            best_j = None
            for j in list(unmatched_det):
                v = iou_xyxy(tr.box, detections[j])
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_j is not None and best_iou >= self.iou_thresh:
                tr.box = detections[best_j]
                tr.last_seen = frame_idx
                tr.missed = 0
                unmatched_det.remove(best_j)
            else:
                tr.missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for j in unmatched_det:
            self.tracks.append(Track(tid=self._next_id, box=detections[j], last_seen=frame_idx))
            self._next_id += 1

        return self.tracks


# ============================
# Detector wrapper (Ultralytics ONNX)
# ============================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, task: str = "detect"):
    return YOLO(model_path, task=task)


def detect_pizzas(
    model,
    frame_bgr: np.ndarray,
    conf: float,
    iou: float,
    pizza_class_id: int,
    imgsz: int = 640,
) -> List[np.ndarray]:
    res = model.predict(
        source=frame_bgr,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        device="cpu",
        verbose=False,
    )

    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    xyxy = r0.boxes.xyxy.cpu().numpy()
    cls = r0.boxes.cls.cpu().numpy().astype(int)

    return [xyxy[k] for k in range(len(xyxy)) if cls[k] == int(pizza_class_id)]


# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Pizza Counter", layout="wide")
st.title("Подсчёт пицц в кадре (ONNX детекция)")

with st.sidebar:
    st.header("Файлы")
    model_file = st.file_uploader("ONNX модель (например best.onnx)", type=["onnx"])
    video_file = st.file_uploader("Видео", type=["mp4", "mov", "avi", "mkv"])

    st.divider()
    st.header("Детекция")
    conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
    nms_iou = st.slider("NMS IoU", 0.1, 0.9, 0.45, 0.05)
    imgsz = st.selectbox("imgsz", [320, 416, 512, 640, 800], index=3)
    pizza_class_id = st.number_input("ID класса 'pizza'", min_value=0, max_value=999, value=0, step=1)

    st.divider()
    st.header("Трекинг (для стабильных ID)")
    trk_iou = st.slider("IOU для трекинга", 0.1, 0.9, 0.3, 0.05)
    max_missed = st.slider("Сколько кадров держать пропавший объект", 1, 200, 20, 1)

    st.divider()
    st.header("Отображение")
    show_every = st.slider("Показывать каждый N-й кадр", 1, 10, 2, 1)

    st.divider()
    st.header("История и отчёты")
    st.caption(f"Файл истории: {HISTORY_PATH}")

process_btn = st.button(
    "Запустить инференс",
    type="primary",
    disabled=(model_file is None or video_file is None),
)

tab1, tab2 = st.tabs(["Инференс", "Статистика / История"])

# ---------
# Tab 1: Inference
# ---------
with tab1:
    col_left, col_right = st.columns([1.35, 1.0], gap="large")

    with col_left:
        st.subheader("Live (bbox + счётчик)")
        live_placeholder = st.empty()

    with col_right:
        st.subheader("Статус")
        progress = st.progress(0)
        status = st.empty()
        st.subheader("Метрики")
        metric_now = st.empty()
        metric_max = st.empty()
        metric_avg = st.empty()

    if process_btn:
        tmp_dir = tempfile.mkdtemp(prefix="pizza_count_")

        model_bytes = model_file.read()
        video_bytes = video_file.read()

        model_path = os.path.join(tmp_dir, "model.onnx")
        with open(model_path, "wb") as f:
            f.write(model_bytes)

        video_path = os.path.join(tmp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        try:
            model = load_model(model_path, task="detect")
        except Exception as e:
            st.error(f"Не удалось загрузить модель: {e}")
            st.stop()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Не удалось открыть видео (VideoCapture).")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        out_path = os.path.join(tmp_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        tracker = SimpleIOUTracker(iou_thresh=float(trk_iou), max_missed=int(max_missed))

        frame_idx = 0
        max_in_frame = 0
        sum_in_frame = 0
        t0 = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                dets = detect_pizzas(
                    model=model,
                    frame_bgr=frame,
                    conf=float(conf),
                    iou=float(nms_iou),
                    pizza_class_id=int(pizza_class_id),
                    imgsz=int(imgsz),
                )

                tracks = tracker.update(dets, frame_idx)

                count_now = len(tracks)
                max_in_frame = max(max_in_frame, count_now)
                sum_in_frame += count_now

                for tr in tracks:
                    x1, y1, x2, y2 = tr.box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID {tr.tid}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    f"Pizzas in frame: {count_now}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                )
                cv2.putText(
                    frame,
                    f"Pizzas in frame: {count_now}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    2,
                )

                writer.write(frame)

                if frame_idx % int(show_every) == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    live_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                frame_idx += 1
                avg_now = round(sum_in_frame / max(1, frame_idx), 3)

                if total_frames > 0:
                    progress.progress(min(1.0, frame_idx / total_frames))
                    status.write(f"Кадр {frame_idx}/{total_frames} | В кадре: {count_now}")
                else:
                    status.write(f"Кадр {frame_idx} | В кадре: {count_now}")

                metric_now.metric("Пицц в кадре сейчас", count_now)
                metric_max.metric("Максимум пицц в кадре", max_in_frame)
                metric_avg.metric("Среднее пицц/кадр", avg_now)

        finally:
            cap.release()
            writer.release()

        processing_sec = round(time.time() - t0, 3)
        avg_in_frame = round(sum_in_frame / max(1, frame_idx), 3)

        status.write(f"Готово. Кадров: {frame_idx}. Время обработки: {processing_sec} сек.")
        st.success("Готово.")

        # Сохраняем запись в историю
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model.name": getattr(model_file, "name", "model.onnx"),
            "video.name": getattr(video_file, "name", "video"),
            "video.fps": round(float(fps), 3),
            "video.frames": int(frame_idx),
            "params.conf": float(conf),
            "params.nms_iou": float(nms_iou),
            "params.imgsz": int(imgsz),
            "params.pizza_class_id": int(pizza_class_id),
            "params.trk_iou": float(trk_iou),
            "params.max_missed": int(max_missed),
            "params.show_every": int(show_every),
            "stats.avg_in_frame": float(avg_in_frame),
            "stats.max_in_frame": int(max_in_frame),
            "stats.processing_sec": float(processing_sec),
        }
        append_history(record)

        st.divider()
        st.subheader("Размеченное видео (результат)")
        with open(out_path, "rb") as f:
            annotated_bytes = f.read()

        st.video(annotated_bytes)

        st.download_button(
            "Скачать размеченное видео",
            data=annotated_bytes,
            file_name="annotated_pizza_count.mp4",
            mime="video/mp4",
        )

# ---------
# Tab 2: Statistics / History / Reports
# ---------
with tab2:
    st.subheader("Статистика и история запусков")

    history = load_history()
    if not history:
        st.info("История пуста. Запустите инференс хотя бы один раз.")
    else:
        df = pd.json_normalize(history)
        df_view = df.sort_values("timestamp", ascending=False)
        st.dataframe(df_view, use_container_width=True, height=420)

        # Быстрая агрегированная статистика
        st.subheader("Сводка (по всем запускам)")
        try:
            avg_of_avg = float(df["stats.avg_in_frame"].mean())
            max_of_max = int(df["stats.max_in_frame"].max())
            total_runs = int(len(df))
            last_ts = str(df_view.iloc[0]["timestamp"])
        except Exception:
            avg_of_avg, max_of_max, total_runs, last_ts = 0.0, 0, len(history), ""

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Запусков", total_runs)
        c2.metric("Среднее (avg_in_frame)", round(avg_of_avg, 3))
        c3.metric("Максимум (max_in_frame)", max_of_max)
        c4.metric("Последний запуск", last_ts)

        st.divider()
        st.subheader("Экспорт")

        # JSON
        json_bytes = json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "Скачать историю (JSON)",
            data=json_bytes,
            file_name="history.json",
            mime="application/json",
        )

        # Excel
        try:
            excel_bytes = generate_excel_bytes(history)
            st.download_button(
                "Скачать отчёт (Excel)",
                data=excel_bytes,
                file_name="pizza_inference_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Не удалось сформировать Excel: {e}")


        st.divider()
        st.subheader("Управление историей")
        if st.button("Очистить историю (удалить history.json)", type="secondary"):
            try:
                if os.path.exists(HISTORY_PATH):
                    os.remove(HISTORY_PATH)
                st.success("История очищена.")
            except Exception as e:
                st.error(f"Не удалось удалить history.json: {e}")
