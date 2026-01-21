# -*- coding: utf-8 -*-
# ============================================================
# 🏊 Freestyle Swimming Technique Analyzer – Streamlit App
# Cloud-safe • Mobile-safe • Coach-ready
# ============================================================
# app.py - Streamlit Web Dashboard for Freestyle Swim Analyzer

# app.py - Streamlit Dashboard: Freestyle Swimming Technique Analyzer

# app.py - Streamlit Dashboard: Freestyle Swimming Technique Analyzer

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
# HELPER FUNCTIONS
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
# STREAMLIT APP
# ────────────────────────────────────────────────

st.set_page_config(page_title="Swim Coach Analyzer", layout="wide")

st.title("Freestyle Swimming Technique Analyzer")
st.markdown("""
Upload a side-view freestyle swimming video to get:
- Annotated video with technique overlays
- CSV export of time-series data
- PDF report with summary, key positions & time-series plot
""")

# Sidebar settings
with st.sidebar:
    st.header("Analysis Settings")
    is_underwater = st.checkbox("Underwater footage", value=False)
    st.caption("Adjust if video is underwater (affects ideal angle ranges)")

uploaded_file = st.file_uploader("Upload video (.mp4 recommended)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.success(f"Video ready: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")

    if st.button("Start Analysis", type="primary"):
        with st.spinner("Processing video... (may take 1–5+ minutes depending on length)"):
            # Temporary file handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(uploaded_file.read())
                input_path = tmp_in.name

            try:
                # ────────────────────────────────────────────────
                # CONFIGURATION
                # ────────────────────────────────────────────────
                SMOOTHING_WINDOW = 7
                ELBOW_MIN_WINDOW = 9
                ELBOW_MIN_PROM = 10.0
                MIN_STROKE_GAP_S = 0.5
                BREATH_SIDE_THRESH = 0.15
                MIN_BREATH_GAP_S = 1.0
                MIN_BREATH_HOLD_FRAMES = 4

                IDEAL_ELBOW = (100, 135)
                IDEAL_KNEE = (120, 160) if is_underwater else (125, 165)
                IDEAL_ROLL_ABS_MAX = 55.0

                # ────────────────────────────────────────────────
                # MODEL DOWNLOAD & SETUP
                # ────────────────────────────────────────────────
                MODEL = "pose_landmarker_heavy.task"
                if not os.path.exists(MODEL):
                    st.info("Downloading MediaPipe model (one-time)...")
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

                # ────────────────────────────────────────────────
                # VIDEO PROCESSING
                # ────────────────────────────────────────────────
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    raise RuntimeError("Cannot open uploaded video")

                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                annotated_path = f"annotated_{timestamp}.mp4"
                writer = cv2.VideoWriter(annotated_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

                # Data storage
                times = []
                elbow_vals = []
                kl_vals = []
                kr_vals = []
                sym_vals = []
                scores = []
                yaw_series = []
                breath_state_series = []
                roll_angles = []

                elbow_buf = deque(maxlen=SMOOTHING_WINDOW)
                kl_buf = deque(maxlen=SMOOTHING_WINDOW)
                kr_buf = deque(maxlen=SMOOTHING_WINDOW)

                elbow_win = deque(maxlen=ELBOW_MIN_WINDOW)
                time_win = deque(maxlen=ELBOW_MIN_WINDOW)
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

                NOSE, LSH, RSH = 0, 11, 12
                LEL, LWR = 13, 15
                LHP, LKN, LAK = 23, 25, 27
                RHP, RKN, RAK = 24, 26, 28

                progress_bar = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_id += 1
                    progress_bar.progress(min(frame_id / 500, 1.0))  # rough estimate

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

                    LS = P(LSH)
                    RS = P(RSH)
                    LE = P(LEL)
                    LW = P(LWR)
                    LH = P(LHP)
                    LK = P(LKN)
                    LA = P(LAK)
                    RH = P(RHP)
                    RK = P(RKN)
                    RA = P(RAK)
                    NZ = P(NOSE)

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

                    elbow_win.append(elbow_s)
                    time_win.append(t_s)
                    if len(elbow_win) == ELBOW_MIN_WINDOW:
                        is_min, _ = local_min_center(list(elbow_win), ELBOW_MIN_PROM)
                        center_t = list(time_win)[ELBOW_MIN_WINDOW // 2]
                        if is_min and (not stroke_times or center_t - stroke_times[-1] >= MIN_STROKE_GAP_S):
                            stroke_times.append(center_t)

                    yaw = yaw_series[-1]
                    desired_side = 'R' if yaw > BREATH_SIDE_THRESH else 'L' if yaw < -BREATH_SIDE_THRESH else 'N'
                    if desired_side == current_side:
                        side_persist_counter += 1
                    else:
                        side_persist_counter = 1
                        current_side = desired_side
                    if current_side in ('L', 'R') and side_persist_counter >= MIN_BREATH_HOLD_FRAMES:
                        if t_s - last_breath_time >= MIN_BREATH_GAP_S:
                            if current_side == 'L':
                                breath_count_L += 1
                            else:
                                breath_count_R += 1
                            last_breath_time = t_s
                    breath_state_series.append(current_side)

                    arm_c   = (0,255,0) if e_dev <= 10 else (0,255,255) if e_dev <= 20 else (0,0,255)
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
                # SUMMARY & OUTPUTS
                # ────────────────────────────────────────────────
                session_duration_s = times[-1] if times else 0.0
                avg_score   = statistics.mean(scores) if scores else 0.0
                avg_sym     = statistics.mean(sym_vals) if sym_vals else 0.0
                avg_roll    = statistics.mean(roll_angles) if roll_angles else 0.0
                max_roll_abs = max(abs(r) for r in roll_angles) if roll_angles else 0.0

                sr_single_avg = sr_both_avg = 0.0
                if len(stroke_times) >= 2:
                    dur = stroke_times[-1] - stroke_times[0]
                    sr_single_avg = 60.0 * (len(stroke_times)-1) / max(dur, 0.1)
                    sr_both_avg = 2.0 * sr_single_avg

                total_min = session_duration_s / 60.0 if session_duration_s > 0 else 1e-6
                bpm_avg = (breath_count_L + breath_count_R) / total_min if total_min > 0.01 else 0.0

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

                # Plot (simplified example - expand as needed)
                fig_buffer = io.BytesIO()
                if len(times) > 5:
                    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                    axs[0].plot(times, elbow_vals, label="Elbow")
                    axs[0].legend(); axs[0].grid(True)
                    axs[1].plot(times, scores, label="Score", color="green")
                    axs[1].legend(); axs[1].grid(True)
                    axs[2].plot(times, roll_angles, label="Roll", color="purple")
                    axs[2].legend(); axs[2].grid(True)
                    axs[3].plot(times, scores, label="Score")
                    for ts in stroke_times:
                        axs[3].axvline(ts, color="cyan", ls="--", alpha=0.5)
                    axs[3].legend(); axs[3].grid(True)
                    plt.savefig(fig_buffer, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    fig_buffer.seek(0)

                # PDF
                pdf_buffer = io.BytesIO()
                pdf = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                styles.add(ParagraphStyle(name='Disclaimer', fontSize=10, textColor=colors.red, spaceAfter=12))
                story = []

                story.append(Paragraph("Freestyle Technique Report", styles["Title"]))
                story.append(Spacer(1,12))
                story.append(Paragraph(f"Video: {uploaded_file.name}", styles["Normal"]))
                story.append(Paragraph(f"Duration: {session_duration_s:.1f} s", styles["Normal"]))
                story.append(Paragraph(f"Avg Score: {avg_score:.1f}/100", styles["Normal"]))
                story.append(Paragraph(f"Avg Roll: {avg_roll:.1f}° (max |roll|: {max_roll_abs:.1f}°)", styles["Normal"]))
                story.append(Spacer(1,12))

                story.append(Paragraph("Summary", styles["Heading2"]))
                story.append(Paragraph(f"Avg Stroke Rate: {sr_single_avg:.1f} / {sr_both_avg:.1f} spm", styles["Normal"]))
                story.append(Paragraph(f"Breaths/min: {bpm_avg:.1f}", styles["Normal"]))
                story.append(Spacer(1,18))

                story.append(Paragraph("Time-Series Plot", styles["Heading2"]))
                if fig_buffer.getvalue():
                    img_plot = RLImage(fig_buffer)
                    img_plot.drawWidth = letter[0] - 72
                    img_plot.drawHeight = (letter[0] - 72) * 0.65
                    story.append(img_plot)
                else:
                    story.append(Paragraph("No plot generated", styles["Italic"]))

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
                        st.download_button(
                            label="Download Annotated Video",
                            data=f,
                            file_name="annotated_swim.mp4",
                            mime="video/mp4"
                        )
                with col2:
                    st.download_button(
                        label="Download CSV",
                        data=csv_buffer,
                        file_name="swim_analysis.csv",
                        mime="text/csv"
                    )
                with col3:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name="swim_report.pdf",
                        mime="application/pdf"
                    )

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.exception(e)
            finally:
                if 'input_path' in locals():
                    os.unlink(input_path)
                if 'annotated_path' in locals() and os.path.exists(annotated_path):
                    os.unlink(annotated_path)  # optional: keep if you want to cache

else:
    st.info("Please upload a video to begin analysis.")
