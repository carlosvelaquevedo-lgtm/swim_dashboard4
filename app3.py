# -*- coding: utf-8 -*-
# ============================================================
# 🏊 Freestyle Swimming Technique Analyzer – Streamlit App
# Cloud-safe • Mobile-safe • Coach-ready
# ============================================================
# app.py - Streamlit Web Dashboard for Freestyle Swim Analyzer

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import datetime
import statistics
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import io
import tempfile
import urllib.request

# ────────────────────────────────────────────────
# Helper functions (same as your Colab)
# ────────────────────────────────────────────────

def angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def deviation(val, ideal_range):
    lo, hi = ideal_range
    if val < lo: return lo - val
    if val > hi: return val - hi
    return 0.0

def local_min_center(window, prom=10.0):
    if len(window) < 3: return False, None
    center = len(window) // 2
    val = window[center]
    left_ok = all(val <= window[i] + prom for i in range(center))
    right_ok = all(val <= window[i] + prom for i in range(center, len(window)))
    return left_ok and right_ok, val

def signed_yaw_proxy(nose, left_shoulder, right_shoulder):
    if right_shoulder[0] == left_shoulder[0]: return 0.0
    dx = right_shoulder[0] - left_shoulder[0]
    expected_nose_x = left_shoulder[0] + dx * 0.5
    return (nose[0] - expected_nose_x) / (abs(dx) + 1e-6)

def shoulder_roll_angle(ls, rs):
    dy = ls[1] - rs[1]
    dx = ls[0] - rs[0]
    if dx == 0: return 90.0 if dy > 0 else -90.0
    return np.degrees(np.arctan2(dy, dx))

# ────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────

st.set_page_config(page_title="Swim Coach Analyzer", layout="wide")

st.title("Freestyle Swimming Technique Analyzer")
st.markdown("Upload a side-view freestyle video to get annotated video, CSV data, and PDF report with analysis.")

uploaded_file = st.file_uploader("Upload video (.mp4 recommended)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.success("Video uploaded! Click **Analyze** to start.")

    if st.button("Analyze Video", type="primary"):
        with st.spinner("Analyzing video... (may take several minutes)"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                input_path = tmp.name

            try:
                # Model download
                MODEL = "pose_landmarker_heavy.task"
                if not os.path.exists(MODEL):
                    urllib.request.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                        MODEL
                    )

                base = python.BaseOptions(model_asset_path=MODEL)
                options = vision.PoseLandmarkerOptions(
                    base_options=base,
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.6,
                    min_pose_presence_confidence=0.6,
                    min_tracking_confidence=0.6
                )
                detector = vision.PoseLandmarker.create_from_options(options)

                # Video setup
                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                annotated_path = f"annotated_{timestamp}.mp4"
                writer = cv2.VideoWriter(annotated_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

                # Data containers (copy from your original)
                times = []
                elbow_vals = []
                kl_vals = []
                kr_vals = []
                sym_vals = []
                scores = []
                yaw_series = []
                breath_state_series = []
                roll_angles = []

                elbow_buf = deque(maxlen=7)
                kl_buf = deque(maxlen=7)
                kr_buf = deque(maxlen=7)

                elbow_win = deque(maxlen=9)
                time_win = deque(maxlen=9)
                stroke_times = []

                last_breath_time = -1e9
                breath_count_L = 0
                breath_count_R = 0
                current_side = 'N'
                side_persist_counter = 0

                best_elbow_dev = float('inf')
                worst_elbow_dev = -float('inf')
                best_frame_bytes = None
                worst_frame_bytes = None
                best_time = None
                worst_time = None

                frame_id = 0

                IDEAL_ELBOW = (100, 135)
                IDEAL_KNEE = (125, 165)

                progress = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_id += 1
                    progress.progress(min(frame_id / 500, 1.0))  # rough

                    orig_frame = frame.copy()
                    h, w = frame.shape[:2]
                    t_s = frame_id / fps
                    t_ms = int(t_s * 1000)

                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = detector.detect_for_video(mp_img, t_ms)

                    if not result.pose_landmarks:
                        writer.write(frame)
                        continue

                    lm = result.pose_landmarks[0]
                    P = lambda i: (int(lm[i].x * w), int(lm[i].y * h))

                    LS = P(11)
                    RS = P(12)
                    LE = P(13)
                    LW = P(15)
                    LH = P(23)
                    LK = P(25)
                    LA = P(27)
                    RH = P(24)
                    RK = P(26)
                    RA = P(28)
                    NZ = P(0)

                    elbow = angle(LS, LE, LW)
                    kl = angle(LH, LK, LA)
                    kr = angle(RH, RK, RA)

                    elbow_buf.append(elbow)
                    kl_buf.append(kl)
                    kr_buf.append(kr)

                    elbow_s = statistics.mean(elbow_buf) if elbow_buf else elbow
                    kl_s = statistics.mean(kl_buf) if kl_buf else kl
                    kr_s = statistics.mean(kr_buf) if kr_buf else kr

                    roll = shoulder_roll_angle(LS, RS)

                    e_dev = deviation(elbow_s, IDEAL_ELBOW)
                    kl_dev = deviation(kl_s, IDEAL_KNEE)
                    kr_dev = deviation(kr_s, IDEAL_KNEE)
                    symmetry = abs(kl_s - kr_s)

                    raw_penalty = e_dev * 0.4 + symmetry * 0.3 + abs(kl_dev - kr_dev) * 0.3
                    score = max(0, min(100, 100 - raw_penalty))

                    times.append(t_s)
                    elbow_vals.append(elbow_s)
                    kl_vals.append(kl_s)
                    kr_vals.append(kr_s)
                    sym_vals.append(symmetry)
                    scores.append(score)
                    yaw_series.append(signed_yaw_proxy(NZ, LS, RS))
                    roll_angles.append(roll)

                    phase = 'Pull' if LW[1] > LS[1] else 'Recovery'
                    if phase == 'Pull':
                        dev = abs(elbow_s - statistics.mean(IDEAL_ELBOW))
                        if dev < best_elbow_dev:
                            best_elbow_dev = dev
                            _, buf = cv2.imencode('.jpg', orig_frame)
                            best_frame_bytes = buf.tobytes()
                            best_time = t_s
                        if dev > worst_elbow_dev:
                            worst_elbow_dev = dev
                            _, buf = cv2.imencode('.jpg', orig_frame)
                            worst_frame_bytes = buf.tobytes()
                            worst_time = t_s

                    # Stroke detection
                    elbow_win.append(elbow_s)
                    time_win.append(t_s)
                    if len(elbow_win) == ELBOW_MIN_WINDOW:
                        is_min, _ = local_min_center(list(elbow_win), ELBOW_MIN_PROM)
                        center_t = list(time_win)[ELBOW_MIN_WINDOW // 2]
                        if is_min and (not stroke_times or center_t - stroke_times[-1] >= MIN_STROKE_GAP_S):
                            stroke_times.append(center_t)

                    # Breathing detection
                    yaw = yaw_series[-1]
                    desired_side = 'R' if yaw > BREATH_SIDE_THRESH else 'L' if yaw < -BREATH_SIDE_THRESH else 'N'
                    if desired_side == current_side:
                        side_persist_counter += 1
                    else:
                        side_persist_counter = 1
                        current_side = desired_side
                    if current_side in ('L', 'R') and side_persist_counter >= MIN_BREATH_HOLD_FRAMES:
                        if t_s - last_breath_time >= MIN_BREATH_GAP_S:
                            if current_side == 'L': breath_count_L += 1
                            else: breath_count_R += 1
                            last_breath_time = t_s
                    breath_state_series.append(current_side)

                    # Draw limbs and text (your original drawing code)
                    arm_c = (0,255,0) if e_dev <= 10 else (0,255,255) if e_dev <= 20 else (0,0,255)
                    leg_l_c = (0,255,0) if kl_dev <= 10 else (0,255,255) if kl_dev <= 20 else (0,0,255)
                    leg_r_c = (0,255,0) if kr_dev <= 10 else (0,255,255) if kr_dev <= 20 else (0,0,255)

                    for a, b, c in [(LS, LE, arm_c), (LE, LW, arm_c),
                                    (LH, LK, leg_l_c), (LK, LA, leg_l_c),
                                    (RH, RK, leg_r_c), (RK, RA, leg_r_c)]:
                        cv2.line(frame, a, b, c, 3)
                        cv2.circle(frame, a, 5, c, -1)
                        cv2.circle(frame, b, 5, c, -1)

                    y = 30
                    cv2.putText(frame, f"Phase: {phase}", (30, y), 0, 0.8, (255,255,255), 2); y += 30
                    cv2.putText(frame, f"Score: {int(score)}", (30, y), 0, 0.8, (0,255,0), 2); y += 30
                    sr_single = 60.0 * (len(stroke_times)-1) / max(1e-6, stroke_times[-1] - stroke_times[0]) if len(stroke_times) >= 2 else 0.0
                    sr_both = 2.0 * sr_single
                    bpm = (breath_count_L + breath_count_R) / max(1e-6, t_s / 60.0)
                    cv2.putText(frame, f"SR: {sr_single:.1f}/{sr_both:.1f} spm", (30, y), 0, 0.7, (255,255,255), 2); y += 25
                    cv2.putText(frame, f"Breaths/min: {bpm:.1f}", (30, y), 0, 0.7, (255,255,255), 2); y += 25
                    cv2.putText(frame, f"Roll: {roll:.1f}°", (30, y), 0, 0.7, (255,255,0), 2)

                    writer.write(frame)

                cap.release()
                writer.release()

                # ────────────────────────────────────────────────
                # POST-PROCESSING
                # ────────────────────────────────────────────────

                # CSV
                df = pd.DataFrame({
                    "time_s": times,
                    "elbow_deg": elbow_vals,
                    "knee_left_deg": kl_vals,
                    "knee_right_deg": kr_vals,
                    "symmetry_deg": sym_vals,
                    "score": scores,
                    "yaw_proxy": yaw_series,
                    "breath_state": breath_state_series,
                    "body_roll_deg": roll_angles
                })
                csv_buffer = io.BytesIO()
                df.to_csv(csv_buffer, index=False, float_format="%.2f")
                csv_buffer.seek(0)

                # Plot (example - add your full plot code here)
                fig_buffer = io.BytesIO()
                if len(times) > 5:
                    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                    # ... (your plot code - axes 0-3)
                    plt.savefig(fig_buffer, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    fig_buffer.seek(0)

                # PDF
                pdf_buffer = io.BytesIO()
                pdf = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                styles.add(ParagraphStyle(name='Disclaimer', fontSize=10, textColor=colors.red, spaceAfter=12))
                story = []

                story.append(Paragraph("Swim Analysis Report", styles["Title"]))
                story.append(Spacer(1,12))
                story.append(Paragraph(f"Video processed: {uploaded_file.name}", styles["Normal"]))
                story.append(Paragraph(f"Avg Score: {avg_score:.1f}/100", styles["Normal"]))
                # ... (add your summary, notes, key frames, plot embedding)

                if fig_buffer.getvalue():
                    story.append(Paragraph("Time-Series Plot", styles["Heading2"]))
                    img_plot = RLImage(fig_buffer)
                    img_plot.drawWidth = letter[0] - 72
                    img_plot.drawHeight = (letter[0] - 72) * 0.65
                    story.append(img_plot)

                pdf.build(story)
                pdf_buffer.seek(0)

                # ────────────────────────────────────────────────
                # DISPLAY RESULTS
                # ────────────────────────────────────────────────
                st.success("Analysis complete!")

                st.video(annotated_path)

                col1, col2, col3 = st.columns(3)
                with col1:
                    with open(annotated_path, "rb") as f:
                        st.download_button("Download Annotated Video", f, file_name="annotated_swim.mp4", mime="video/mp4")

                with col2:
                    st.download_button("Download CSV", csv_buffer, "swim_analysis.csv", mime="text/csv")

                with col3:
                    st.download_button("Download PDF Report", pdf_buffer, "swim_report.pdf", mime="application/pdf")

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
            finally:
                if os.path.exists(input_path):
                    os.unlink(input_path)

else:
    st.info("Upload a video to begin.")
