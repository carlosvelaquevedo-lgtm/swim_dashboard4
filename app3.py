# -*- coding: utf-8 -*-
# =========================================================
# 🏊 Freestyle Swimming Technique Analyzer — Streamlit App
# =========================================================

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import zipfile
import os
import io
import math
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from fpdf import FPDF

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
mp_pose = mp.solutions.pose

# ---------------- UTILS ----------------
def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def safe_get(lm, idx):
    return np.array([lm[idx].x, lm[idx].y])

# ---------------- PDF ----------------
class Report(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Freestyle Technique Report", ln=True)

def generate_pdf(name, strokes, score):
    pdf = Report()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Athlete / Video: {name}", ln=True)
    pdf.cell(0, 8, f"Technique Score: {score:.1f}/100", ln=True)
    pdf.ln(4)

    for i, s in enumerate(strokes, 1):
        pdf.cell(0, 7, f"Stroke {i}", ln=True)
        pdf.cell(0, 7, f"  Avg Elbow: {s['elbow']:.1f}°", ln=True)
        pdf.cell(0, 7, f"  Avg Knee: {s['knee']:.1f}°", ln=True)
        pdf.cell(0, 7, f"  Symmetry Error: {s['sym']:.3f}", ln=True)
        pdf.ln(2)

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(out.name)
    return out.name

# ---------------- STREAMLIT UI ----------------
st.title("🏊 Freestyle Swimming Technique Analyzer")

uploaded = st.file_uploader(
    "Upload freestyle swimming videos",
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True
)

with st.sidebar:
    st.header("Coach Settings")
    underwater = st.toggle("Underwater footage")
    strict = st.toggle("Strict scoring")
    preview = st.toggle("Show preview frames")
    frame_skip = st.slider("Frame skip (mobile)", 0, 3, 1)
    smooth_win = st.slider("Angle smoothing", 3, 15, 7)

# ---------------- PROCESS ----------------
results = []
zip_files = []

if uploaded:
    for file in uploaded:
        st.subheader(file.name)
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())
        cap = cv2.VideoCapture(tmp.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 10 or fps > 120:
            fps = 30

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        elbows, knees, sym = [], [], []
        elbow_q = deque(maxlen=smooth_win)
        knee_q = deque(maxlen=smooth_win)

        timestamp = 0
        frame_id = 0
        stroke_bins = defaultdict(list)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % (frame_skip + 1) != 0:
                continue

            timestamp += int(1000 / fps)

            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img)
                if not res.pose_landmarks:
                    continue

                lm = res.pose_landmarks.landmark

                sh = safe_get(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                el = safe_get(lm, mp_pose.PoseLandmark.LEFT_ELBOW.value)
                wr = safe_get(lm, mp_pose.PoseLandmark.LEFT_WRIST.value)

                hip = safe_get(lm, mp_pose.PoseLandmark.LEFT_HIP.value)
                kn = safe_get(lm, mp_pose.PoseLandmark.LEFT_KNEE.value)
                an = safe_get(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value)

                ea = angle(sh, el, wr)
                ka = angle(hip, kn, an)

                elbow_q.append(ea)
                knee_q.append(ka)

                ea_s = np.mean(elbow_q)
                ka_s = np.mean(knee_q)

                elbows.append(ea_s)
                knees.append(ka_s)

                sym_err = abs(ea_s - ka_s) / 180
                sym.append(sym_err)

                stroke_idx = int(timestamp / 1200)
                stroke_bins[stroke_idx].append((ea_s, ka_s, sym_err))

                if preview and frame_id % 90 == 0:
                    st.image(frame, channels="BGR", width=300)

            except Exception:
                continue  # graceful MediaPipe failure recovery

        cap.release()

        strokes = []
        for k, v in stroke_bins.items():
            if len(v) < 3:
                continue
            e = np.mean([x[0] for x in v])
            k_ = np.mean([x[1] for x in v])
            s = np.mean([x[2] for x in v])
            strokes.append({"elbow": e, "knee": k_, "sym": s})

        base = np.mean(sym) * 100 if sym else 50
        score = max(0, 100 - base * (1.3 if strict else 1.0))

        pdf_path = generate_pdf(file.name, strokes, score)
        zip_files.append(pdf_path)

        # Charts
        fig, ax = plt.subplots()
        ax.plot(elbows, label="Elbow")
        ax.plot(knees, label="Knee")
        ax.legend()
        st.pyplot(fig)

        results.append((file.name, score))

    # ---------------- COMPARISON ----------------
    st.header("📊 Coach Comparison View")
    names = [r[0] for r in results]
    scores = [r[1] for r in results]

    fig, ax = plt.subplots()
    ax.bar(names, scores)
    ax.set_ylabel("Technique Score")
    st.pyplot(fig)

    # ---------------- ZIP ----------------
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in zip_files:
            z.write(f, arcname=os.path.basename(f))

    with open(zip_path, "rb") as f:
        st.download_button(
            "⬇ Download ALL Results (ZIP)",
            data=f,
            file_name="swim_analysis_results.zip",
            mime="application/zip"
        )
