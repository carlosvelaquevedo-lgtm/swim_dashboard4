# -*- coding: utf-8 -*-
# ============================================================
# 🏊 Freestyle Swimming Technique Analyzer – Streamlit App
# Cloud-safe • Mobile-safe • Coach-ready
# ============================================================

import streamlit as st
import cv2, os, zipfile, tempfile, urllib.request, statistics
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from fpdf import FPDF
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from datetime import datetime

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="Freestyle Swimming Analyzer", layout="wide")
st.title("🏊 Freestyle Swimming Technique Analyzer")

# ============================================================
# COACH SETTINGS
# ============================================================
st.sidebar.header("Coach Settings")

UNDERWATER = st.sidebar.toggle("Underwater footage", False)
STRICT = st.sidebar.toggle("Strict scoring", False)
SHOW_PREVIEW = st.sidebar.toggle("Show preview frames", False)
FRAME_SKIP = st.sidebar.slider("Frame skip (mobile)", 0, 3, 1)
SMOOTH = st.sidebar.slider("Angle smoothing", 3, 15, 7)

PENALTY_MULT = 1.3 if STRICT else 1.0
IDEAL_ELBOW = (70,120) if UNDERWATER else (80,140)
IDEAL_KNEE  = (150,175)

# ============================================================
# HELPERS
# ============================================================
def angle(a,b,c):
    ba, bc = np.array(a)-np.array(b), np.array(c)-np.array(b)
    cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def deviation(v,r):
    return max(r[0]-v,0)+max(v-r[1],0)

# ============================================================
# MEDIAPIPE TASKS MODEL
# ============================================================
MODEL = "pose_landmarker_heavy.task"
if not os.path.exists(MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        MODEL
    )

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

# ============================================================
# FILE UPLOAD
# ============================================================
videos = st.file_uploader(
    "Upload freestyle swimming videos",
    type=["mp4","mov","avi"],
    accept_multiple_files=True
)

if not videos:
    st.stop()

results_summary = []
temp_dir = tempfile.mkdtemp()
zip_path = os.path.join(temp_dir, "swim_results.zip")
zipf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

# ============================================================
# PROCESS VIDEOS
# ============================================================
for video in videos:
    st.subheader(video.name)
    video_path = os.path.join(temp_dir, video.name)
    with open(video_path,"wb") as f:
        f.write(video.read())

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w,h = int(cap.get(3)), int(cap.get(4))

    elbow_buf = deque(maxlen=SMOOTH)
    scores, sym = [], []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % (FRAME_SKIP+1) != 0:
            continue

        try:
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            res = detector.detect_for_video(mp_img, int(frame_id/fps*1000))
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks[0]
            P = lambda i: (int(lm[i].x*w), int(lm[i].y*h))

            LS,LE,LW = P(11),P(13),P(15)
            LH,LK,LA = P(23),P(25),P(27)
            RH,RK,RA = P(24),P(26),P(28)

            elbow = angle(LS,LE,LW)
            kneeL = angle(LH,LK,LA)
            kneeR = angle(RH,RK,RA)

            elbow_buf.append(elbow)
            elbow_s = statistics.mean(elbow_buf)

            penalty = (
                deviation(elbow_s,IDEAL_ELBOW)*0.4 +
                abs(kneeL-kneeR)*0.4 +
                (deviation(kneeL,IDEAL_KNEE)+deviation(kneeR,IDEAL_KNEE))*0.2
            ) * PENALTY_MULT

            score = max(0,100-penalty)
            scores.append(score)
            sym.append(abs(kneeL-kneeR))

            if SHOW_PREVIEW and frame_id % 90 == 0:
                st.image(frame, channels="BGR")

        except Exception:
            continue   # ✅ graceful MediaPipe failure recovery

    cap.release()

    avg_score = round(statistics.mean(scores),1) if scores else 0
    avg_sym = round(statistics.mean(sym),1) if sym else 0

    results_summary.append((video.name, avg_score, avg_sym))

    # ------------------ CHART ------------------
    fig, ax = plt.subplots()
    ax.plot(scores, label="Score")
    ax.plot(sym, label="Symmetry")
    ax.legend()
    chart_path = f"{video.name}_chart.png"
    fig.savefig(chart_path)
    plt.close(fig)
    zipf.write(chart_path)

    # ------------------ PDF ------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,video.name,ln=1)
    pdf.set_font("Arial","",11)
    pdf.cell(0,8,f"Average Score: {avg_score}",ln=1)
    pdf.cell(0,8,f"Average Symmetry: {avg_sym}",ln=1)
    pdf.image(chart_path,w=170)

    pdf_path = f"{video.name}.pdf"
    pdf.output(pdf_path)
    zipf.write(pdf_path)

# ============================================================
# COACH COMPARISON TABLE
# ============================================================
st.subheader("📊 Coach Comparison")
st.table(
    [{"Video":v,"Avg Score":s,"Symmetry":y} for v,s,y in results_summary]
)

zipf.close()

# ============================================================
# DOWNLOAD
# ============================================================
with open(zip_path,"rb") as f:
    st.download_button(
        "⬇ Download ALL Results (ZIP)",
        data=f,
        file_name="swim_analysis_results.zip",
        mime="application/zip"
    )

