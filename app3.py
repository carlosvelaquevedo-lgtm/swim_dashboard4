# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile, os, datetime, csv, statistics, urllib.request, zipfile
from collections import deque
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# ===============================================================
# STREAMLIT CONFIG
# ===============================================================
st.set_page_config("Freestyle Swimming Technique Analyzer", layout="wide")
st.title("🏊 Freestyle Swimming Technique Analyzer")

# ===============================================================
# SIDEBAR — COACH CONTROLS
# ===============================================================
with st.sidebar:
    st.header("Coach Settings")
    IS_UNDERWATER = st.toggle("Underwater footage", False)
    STRICT_MODE = st.toggle("Strict scoring", False)
    SHOW_PREVIEW = st.toggle("Show preview frames", False)
    FRAME_SKIP = st.slider("Frame skip (mobile)", 0, 3, 1)
    SMOOTHING_WINDOW = st.slider("Angle smoothing", 3, 15, 7)

# ===============================================================
# CONSTANTS
# ===============================================================
IDEAL_ELBOW = (70,120) if IS_UNDERWATER else (80,140)
IDEAL_KNEE  = (150,175) if IS_UNDERWATER else (155,175)
PENALTY_MULT = 1.3 if STRICT_MODE else 1.0
ELBOW_MIN_PROM = 10
MIN_STROKE_GAP_S = 0.5

# ===============================================================
# HELPERS
# ===============================================================
def angle(a,b,c):
    ba, bc = np.array(a)-np.array(b), np.array(c)-np.array(b)
    return np.degrees(np.arccos(
        np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6),-1,1)
    ))

def deviation(v, r):
    return max(r[0]-v, v-r[1], 0)

def local_min_center(win, prom):
    c = len(win)//2
    v = win[c]
    return all(v <= win[i] + prom for i in range(len(win)))

# ===============================================================
# MEDIAPIPE — SAFE + CACHED
# ===============================================================
@st.cache_resource
def load_detector():
    model = "pose_landmarker_lite.task"
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    if not os.path.exists(model):
        urllib.request.urlretrieve(url, model)
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(opts)

detector = load_detector()

# ===============================================================
# VIDEO UPLOAD
# ===============================================================
uploads = st.file_uploader(
    "Upload freestyle swimming videos",
    type=["mp4","mov","avi"],
    accept_multiple_files=True
)

if not uploads or not st.button("▶ Run Analysis"):
    st.stop()

progress = st.progress(0.0)
coach_summary = []

# ===============================================================
# PROCESS VIDEOS
# ===============================================================
for vid_idx, upload in enumerate(uploads):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upload.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 10 or fps > 120:
        fps = 30  # WhatsApp fix

    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"video_{vid_idx}_{ts}"
    out_vid, out_csv, out_pdf = f"{base}.mp4", f"{base}.csv", f"{base}.pdf"

    writer = cv2.VideoWriter(
        out_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h)
    )

    times, elbows, symm, scores = [], [], [], []
    stroke_times, stroke_scores = [], []

    elbow_buf = deque(maxlen=SMOOTHING_WINDOW)
    elbow_win = deque(maxlen=9)
    time_win = deque(maxlen=9)

    video_ts_ms = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % (FRAME_SKIP+1) != 0:
            continue

        video_ts_ms += int(1000 / fps)
        t = video_ts_ms / 1000

        try:
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            res = detector.detect_for_video(mp_img, video_ts_ms)
        except Exception:
            writer.write(frame)
            continue

        if not res.pose_landmarks:
            writer.write(frame)
            continue

        lm = res.pose_landmarks[0]
        P = lambda i: (int(lm[i].x*w), int(lm[i].y*h))

        LS, LE, LW = P(11), P(13), P(15)
        LH, LK, LA = P(23), P(25), P(27)
        RH, RK, RA = P(24), P(26), P(28)

        e = angle(LS,LE,LW)
        kl = angle(LH,LK,LA)
        kr = angle(RH,RK,RA)

        elbow_buf.append(e)
        e_s = statistics.mean(elbow_buf)
        symmetry = abs(kl-kr)

        penalty = (
            deviation(e_s, IDEAL_ELBOW)*0.4 +
            symmetry*0.4 +
            (deviation(kl,IDEAL_KNEE)+deviation(kr,IDEAL_KNEE))*0.2
        ) * PENALTY_MULT

        score = max(0,100-penalty)

        times.append(t)
        elbows.append(e_s)
        symm.append(symmetry)
        scores.append(score)

        elbow_win.append(e_s)
        time_win.append(t)
        if len(elbow_win) == elbow_win.maxlen:
            if local_min_center(list(elbow_win), ELBOW_MIN_PROM):
                if not stroke_times or t - stroke_times[-1] > MIN_STROKE_GAP_S:
                    stroke_times.append(t)
                    stroke_scores.append(score)

        cv2.putText(frame,f"Score {int(score)}",(20,40),0,0.9,(0,255,0),2)
        writer.write(frame)

        progress.progress((vid_idx + frame_id/total_frames)/len(uploads))

    cap.release()
    writer.release()
    os.remove(video_path)

    # ===========================================================
    # CSV
    # ===========================================================
    with open(out_csv,"w",newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["time","elbow","symmetry","score"])
        for i in range(len(times)):
            wcsv.writerow([
                round(times[i],2),
                round(elbows[i],1),
                round(symm[i],1),
                round(scores[i],1)
            ])

    # ===========================================================
    # CHARTS
    # ===========================================================
    fig, ax = plt.subplots()
    ax.plot(times, scores)
    ax.set_title("Technique Score Over Time")
    score_plot = f"{base}_score.png"
    fig.savefig(score_plot)
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(times, symm)
    ax2.set_title("Symmetry Over Time")
    sym_plot = f"{base}_sym.png"
    fig2.savefig(sym_plot)
    plt.close(fig2)

    # ===========================================================
    # PDF REPORT (PER-STROKE)
    # ===========================================================
    doc = SimpleDocTemplate(out_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("<b>Freestyle Technique Report</b>", styles["Title"]),
        Spacer(1,12),
        Paragraph(f"Average Score: {round(statistics.mean(scores),1)}", styles["Normal"]),
        Paragraph(f"Stroke Count: {len(stroke_times)}", styles["Normal"]),
        Spacer(1,12),
        RLImage(score_plot, width=400, height=200),
        Spacer(1,12),
        RLImage(sym_plot, width=400, height=200),
        Spacer(1,12),
        Paragraph("Per-Stroke Breakdown", styles["Heading2"])
    ]

    table_data = [["Stroke #","Time (s)","Score"]]
    for i,(t_s,s_s) in enumerate(zip(stroke_times,stroke_scores)):
        table_data.append([i+1, round(t_s,2), round(s_s,1)])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("GRID",(0,0),(-1,-1),0.5,colors.grey)
    ]))
    elements.append(table)
    doc.build(elements)

    coach_summary.append({
        "Video": upload.name,
        "Avg Score": round(statistics.mean(scores),1),
        "Strokes": len(stroke_times),
        "Avg Stroke Score": round(statistics.mean(stroke_scores),1) if stroke_scores else 0
    })

# ===============================================================
# COACH COMPARISON VIEW
# ===============================================================
st.subheader("📊 Coach Comparison Across Videos")
st.dataframe(coach_summary)

# ===============================================================
# ZIP DOWNLOAD
# ===============================================================
zip_name = "swim_analysis_results.zip"
with zipfile.ZipFile(zip_name,"w") as z:
    for f in os.listdir():
        if f.endswith((".mp4",".csv",".pdf",".png")):
            z.write(f)

with open(zip_name,"rb") as f:
    st.download_button("⬇ Download ALL Results (ZIP)", f)

st.success("Analysis complete!")
