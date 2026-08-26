import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import time

# ── import fitur extractor dari util ──────────────────────────────────────────
from util.angles import (
    extract_curl_features,
    extract_squat_features,
    extract_pushup_features,
    get_pose_validity,
    get_angle_zone,
)

# ── Konstanta ──────────────────────────────────────────────────────────────────
DATASET_FILES = {
    "curl":   "dataset_curl.csv",
    "squat":  "dataset_squat.csv",
    "pushup": "dataset_pushup.csv",
}

LABELS     = ["bad", "good", "perfect"]
EXERCISES  = ["curl", "squat", "pushup"]  # SEMUA bisa direkam & diedit

# Mapping untuk display name
EXERCISE_NAMES = {
    "curl":   "💪 Dumbbell Curl (Side View)",
    "squat":  "🏋️ Squat (Side View)",
    "pushup": "🤸 Push-Up (Front View)",
}

# Zona sudut untuk setiap olahraga
ZONE_THRESHOLDS = {
    "curl":   {"bad": 160, "good": 100, "perfect": 55},
    "squat":  {"bad": 160, "good": 105, "perfect": 90},
    "pushup": {"bad": 160, "good": 110, "perfect": 80},
}

ZONE_COLOR = {
    "bad":     (80,  80,  255),   # merah
    "good":    (0,   200, 255),   # kuning-biru
    "perfect": (0,   255, 136),   # hijau
}

ZONE_HINT = {
    "curl": {
        "bad":     "Lengan lurus (>160°) — tahan di sini",
        "good":    "Setengah curl (100-160°) — gerak pelan",
        "perfect": "Puncak curl (<100°) — kontraksi penuh",
    },
    "squat": {
        "bad":     "Berdiri tegak (>160°) — tahan di sini",
        "good":    "Setengah squat (100-160°) — turun pelan",
        "perfect": "Squat penuh (<100°) — tahan di bawah",
    },
    "pushup": {
        "bad":     "Lengan lurus (>160°) — posisi atas pushup",
        "good":    "Setengah turun (100-160°) — gerak pelan",
        "perfect": "Dada menyentuh lantai / siku <100°",
    },
}

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

st.set_page_config(
    page_title="FitMove — Data Collection (Curl, Squat, Pushup)",
    page_icon="🎯",
    layout="wide",
)

# Inisialisasi session state
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'countdown_active' not in st.session_state:
    st.session_state.countdown_active = False
if 'countdown_value' not in st.session_state:
    st.session_state.countdown_value = 5


def refresh_data():
    """Force refresh data dari file CSV"""
    st.session_state.refresh_counter += 1
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: baca dataset yang sudah ada (dengan cache control & handle empty file)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=0)
def load_dataset_cached(exercise: str, _refresh):
    """
    Load dataset dengan cache yang bisa di-refresh.
    Handle file kosong dengan aman (return empty DataFrame).
    """
    path = DATASET_FILES.get(exercise)
    if path and os.path.exists(path):
        # Cek ukuran file, jika 0 byte return empty dataframe
        if os.path.getsize(path) == 0:
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(path, header=None)
            # Jika dataframe kosong (tidak ada baris), return empty
            if df.empty:
                return pd.DataFrame()
            return df
        except pd.errors.EmptyDataError:
            # File kosong atau tidak bisa dibaca
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"⚠️ Gagal membaca {path}: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_dataset(exercise: str) -> pd.DataFrame:
    """Load dataset dengan refresh counter."""
    return load_dataset_cached(exercise, st.session_state.refresh_counter)

def count_labels(exercise: str) -> dict:
    """Hitung jumlah sample per label, handle empty dataset."""
    df = load_dataset(exercise)
    if df.empty:
        return {l: 0 for l in LABELS}
    
    # Cek apakah kolom label ada (kolom terakhir)
    if df.shape[1] == 0:
        return {l: 0 for l in LABELS}
    
    try:
        counts = df.iloc[:, -1].value_counts().to_dict()
        return {l: counts.get(l, 0) for l in LABELS}
    except Exception:
        return {l: 0 for l in LABELS}

def append_rows(exercise: str, rows: list):
    """Tambahkan baris baru ke CSV (tanpa header)."""
    path = DATASET_FILES[exercise]
    new_df = pd.DataFrame(rows)
    
    # Jika file tidak ada atau kosong, tulis dengan mode 'w'
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        new_df.to_csv(path, mode="w", header=False, index=False)
    else:
        new_df.to_csv(path, mode="a", header=False, index=False)

def delete_label_data(exercise: str, label: str):
    """Hapus data dengan label tertentu dari dataset."""
    path = DATASET_FILES[exercise]
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df_all = pd.read_csv(path, header=None)
            if df_all.empty:
                return 0
            before = len(df_all)
            df_all = df_all[df_all.iloc[:, -1] != label]
            after = len(df_all)
            
            if after == 0:
                # Jika semua data dihapus, hapus file
                os.remove(path)
            else:
                df_all.to_csv(path, header=False, index=False)
            return before - after
        except Exception:
            return 0
    return 0

def delete_all_data(exercise: str):
    """Hapus semua data untuk exercise tertentu."""
    path = DATASET_FILES[exercise]
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def fix_empty_files():
    """Perbaiki file CSV yang kosong dengan menghapusnya."""
    for exercise, path in DATASET_FILES.items():
        if os.path.exists(path) and os.path.getsize(path) == 0:
            os.remove(path)
            st.info(f"🗑️ File kosong {path} telah dihapus.")


# Panggil fungsi fix di awal untuk membersihkan file kosong
fix_empty_files()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: gambar sudut pada frame
# ──────────────────────────────────────────────────────────────────────────────
def draw_angle_arc(frame, angle: float, zone: str, exercise: str):
    h, w = frame.shape[:2]
    color = ZONE_COLOR.get(zone, (180, 180, 180))
    label_map = {"bad": "BAD", "good": "GOOD", "perfect": "PERFECT"}

    cv2.rectangle(frame, (0, 0), (260, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (260, 100), color, 2)
    cv2.putText(frame, f"{angle:.1f} deg", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, label_map.get(zone, ""), (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def draw_skeleton(frame, landmarks, exercise: str):
    if exercise == "curl":
        color = (0, 212, 255)
    elif exercise == "squat":
        color = (53, 107, 255)
    else:
        color = (0, 255, 136)  # pushup: hijau
    mp_drawing.draw_landmarks(
        frame, landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1),
    )

def get_extract_function(exercise):
    """Dapatkan fungsi ekstraksi fitur yang sesuai."""
    if exercise == "curl":
        return extract_curl_features
    elif exercise == "squat":
        return extract_squat_features
    else:
        return extract_pushup_features

def get_angle_from_features(features, exercise):
    """Dapatkan sudut utama dari features."""
    if exercise == "curl":
        return features[0]  # elbow angle
    elif exercise == "squat":
        return features[0]  # knee angle
    else:
        # Pushup: rata-rata sudut siku kiri dan kanan
        return (features[0] + features[1]) / 2


# ──────────────────────────────────────────────────────────────────────────────
# UI Utama
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:sans-serif;font-size:2rem;font-weight:800;
           margin-bottom:0'>🎯 FitMove — Data Collection</h1>
<p style='color:#888;margin-top:4px'>
  Kumpulkan data latih untuk model <strong style='color:#00d4ff'>CURL</strong>, 
  <strong style='color:#ff6b35'>SQUAT</strong>, dan <strong style='color:#00ff88'>PUSHUP</strong>
</p>
""", unsafe_allow_html=True)

st.divider()

# Tombol Refresh Data
col_refresh1, col_refresh2, col_refresh3 = st.columns([5, 1, 1])
with col_refresh2:
    if st.button("🔄 Refresh Data", help="Refresh tampilan data dari file CSV", use_container_width=True):
        refresh_data()
with col_refresh3:
    if st.button("🗑️ Bersihkan File Kosong", help="Hapus file CSV yang kosong", use_container_width=True):
        fix_empty_files()
        refresh_data()

# ── Sidebar: Pilihan & Statistik ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    st.markdown("---")
    
    exercise = st.selectbox(
        "Pilih Olahraga",
        options=EXERCISES,
        format_func=lambda x: EXERCISE_NAMES.get(x, x),
    )
    
    label = st.selectbox(
        "Label yang Direkam",
        options=LABELS,
        format_func=lambda x: {
            "bad":     "🔴 BAD",
            "good":    "🟡 GOOD",
            "perfect": "🟢 PERFECT",
        }[x],
    )
    
    # Tambahan: durasi countdown
    countdown_duration = st.slider(
        "⏱️ Countdown sebelum rekam (detik)",
        min_value=2, max_value=10, value=5,
        help="Jeda waktu setelah klik MULAI REKAM sebelum recording dimulai"
    )
    
    sample_rate = st.slider(
        "Frekuensi sampel (frame/detik)",
        min_value=5, max_value=30, value=15,
        help="Berapa banyak frame yang disimpan per detik saat merekam",
    )

    st.divider()
    
    # Statistik dataset
    st.markdown(f"### 📊 Statistik Dataset {exercise.upper()}")
    counts = count_labels(exercise)
    total  = sum(counts.values())
    st.markdown(f"**Total data {exercise}:** `{total}` baris")
    for lbl in LABELS:
        emoji = {"bad": "🔴", "good": "🟡", "perfect": "🟢"}[lbl]
        bar   = "█" * min(counts[lbl] // 5, 20) if counts[lbl] > 0 else ""
        st.markdown(
            f"{emoji} **{lbl.upper()}** — `{counts[lbl]}`  \n"
            f"<span style='font-family:monospace;color:#555'>{bar}</span>",
            unsafe_allow_html=True,
        )
    
    target = "≥150" if exercise == "curl" else "≥100"
    st.caption(f"Target ideal: {target} per label untuk {exercise}")

    st.divider()
    st.markdown("### 💡 Tips Rekaman")
    if exercise == "pushup":
        st.markdown("""
        **Push-Up (Front View):**
        1. Menghadap **langsung ke kamera**
        2. Pastikan seluruh badan terlihat
        3. Turun hingga dada dekat lantai
        4. Rekam 1 label sekaligus
        """)
    else:
        st.markdown("""
        **Curl / Squat (Side View):**
        1. Berdiri **dari samping**, menghadap kanan
        2. Pastikan seluruh badan terlihat
        3. Lakukan gerakan **pelan dan terkontrol**
        4. Tahan di zona yang sesuai label
        """)

    st.divider()

    # Tombol hapus data
    with st.expander("🗑️ Hapus Data"):
        st.warning(f"⚠️ Hapus data untuk {exercise.upper()}")
        
        del_label = st.selectbox("Pilih label yang akan dihapus", LABELS, key="del_lbl")
        
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button(f"🗑️ Hapus Label {del_label.upper()}", type="secondary"):
                deleted = delete_label_data(exercise, del_label)
                if deleted > 0:
                    st.success(f"✅ Dihapus {deleted} baris data {del_label} ({exercise})")
                    refresh_data()
                else:
                    st.info(f"Tidak ada data {del_label} untuk dihapus")
        
        with col_del2:
            if st.button("🗑️ Hapus SEMUA Data", type="secondary"):
                if delete_all_data(exercise):
                    st.success(f"✅ Semua data {exercise} dihapus!")
                    refresh_data()
                else:
                    st.info(f"Tidak ada data {exercise} untuk dihapus")


# ── Area utama: panduan posisi ────────────────────────────────────────────────
col_info, col_cam = st.columns([1, 2])

with col_info:
    st.markdown("### 📋 Panduan Posisi")
    hint = ZONE_HINT[exercise][label]
    thresholds = ZONE_THRESHOLDS[exercise]
    
    color_hex = {"bad": "#ff5555", "good": "#ffaa00", "perfect": "#00ff88"}[label]
    st.markdown(f"""
    <div style='background:#111;border-radius:12px;padding:16px;
                border:2px solid {color_hex};margin-bottom:16px'>
      <div style='color:{color_hex};font-weight:700;font-size:.85rem;
                  text-transform:uppercase;letter-spacing:.08em'>
        Label: {label.upper()}
      </div>
      <div style='margin-top:8px;font-size:.9rem;color:#ccc'>
        {hint}
      </div>
      <div style='margin-top:12px;font-size:.8rem;color:#888'>
        Sudut target: 
        {f"< {thresholds['perfect']}°" if label == 'perfect' 
         else f"{thresholds['good']}° - {thresholds['bad']}°" if label == 'good'
         else f"> {thresholds['bad']}°"}
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Zona sudut real-time:**")
    st.markdown(f"🔴 BAD: > {thresholds['bad']}°")
    st.markdown(f"🟡 GOOD: {thresholds['good']}° - {thresholds['bad']}°")
    st.markdown(f"🟢 PERFECT: < {thresholds['perfect']}°")

with col_cam:
    frame_ph = st.empty()
    status_ph = st.empty()
    countdown_ph = st.empty()
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        start_btn = st.button(
            "▶ MULAI REKAM",
            type="primary",
            use_container_width=True,
            key="btn_start",
        )
    with ctrl_col2:
        stop_placeholder = st.empty()
    with ctrl_col3:
        stat_ph = st.empty()


# ── Sesi rekaman dengan COUNTDOWN ─────────────────────────────────────────────
if "recording" not in st.session_state:
    st.session_state.recording = False
if "recording_active" not in st.session_state:
    st.session_state.recording_active = False
if "collected_this_session" not in st.session_state:
    st.session_state.collected_this_session = 0

if start_btn:
    st.session_state.recording = True
    st.session_state.recording_active = False
    st.session_state.collected_this_session = 0
    st.session_state.countdown_value = countdown_duration

if st.session_state.recording:
    pose = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(0)
    
    extract_fn = get_extract_function(exercise)
    rows_buffer = []
    last_sample_time = 0.0
    sample_interval = 1.0 / sample_rate
    
    stop_btn = stop_placeholder.button(
        "⏹ STOP",
        type="secondary",
        use_container_width=True,
        key="btn_stop",
    )
    
    # ========== COUNTDOWN SEBELUM MULAI REKAM ==========
    if not st.session_state.recording_active:
        for i in range(st.session_state.countdown_value, 0, -1):
            # Tampilkan countdown di frame
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Background hitam transparan
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2 - 150, h//2 - 100), (w//2 + 150, h//2 + 100), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                # Tampilkan angka countdown besar
                cv2.putText(frame, str(i), (w//2 - 40, h//2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 136), 8)
                cv2.putText(frame, "BERSIAP...", (w//2 - 100, h//2 - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Tampilkan instruksi posisi
                if exercise == "pushup":
                    cv2.putText(frame, "Hadap KAMERA, posisi pushup", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
                else:
                    cv2.putText(frame, "Hadap SAMPING (kanan/kiri)", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
                
                frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                countdown_ph.markdown(f"<h2 style='text-align:center;color:#00ff88'>⏱️ Mulai dalam {i} detik...</h2>", unsafe_allow_html=True)
                
                time.sleep(1)
        
        countdown_ph.empty()
        st.session_state.recording_active = True
        status_ph.info("🔴 REKAMAN DIMULAI! Lakukan gerakan...")
    
    # ========== PROSES REKAMAN ==========
    try:
        while st.session_state.recording_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            
            current_time = time.time()
            zone_now = None
            angle_now = 0.0
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                valid, reason = get_pose_validity(lm, exercise)
                
                if valid:
                    draw_skeleton(frame, res.pose_landmarks, exercise)
                    features = extract_fn(lm)
                    angle_now = get_angle_from_features(features, exercise)
                    
                    # Tentukan zona berdasarkan sudut
                    thresholds = ZONE_THRESHOLDS[exercise]
                    if angle_now >= thresholds["bad"]:
                        zone_now = "bad"
                    elif angle_now >= thresholds["good"]:
                        zone_now = "good"
                    else:
                        zone_now = "perfect"
                    
                    draw_angle_arc(frame, angle_now, zone_now, exercise)
                    
                    # Rekam sampel sesuai frekuensi
                    if current_time - last_sample_time >= sample_interval:
                        row = features + [label]
                        rows_buffer.append(row)
                        last_sample_time = current_time
                        
                        if len(rows_buffer) >= 30:
                            append_rows(exercise, rows_buffer)
                            st.session_state.collected_this_session += len(rows_buffer)
                            rows_buffer = []
                else:
                    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 60), -1)
                    cv2.putText(frame, reason, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2)
            else:
                cv2.putText(frame, "Pose tidak terdeteksi", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            
            # Indikator RECORDING
            if int(current_time * 2) % 2 == 0:
                cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 70, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            
            # Label yang direkam
            lbl_color = {"bad": (80,80,255), "good": (0,200,255), "perfect": (0,255,136)}[label]
            cv2.putText(frame, f"Label: {label.upper()}", (w - 200, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lbl_color, 2)
            
            # Info olahraga
            cv2.putText(frame, f"Recording: {exercise.upper()}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            total_collected = st.session_state.collected_this_session + len(rows_buffer)
            stat_ph.metric("Direkam sesi ini", total_collected)
            
            if stop_btn:
                break
    
    finally:
        if rows_buffer:
            append_rows(exercise, rows_buffer)
            st.session_state.collected_this_session += len(rows_buffer)
        
        cap.release()
        pose.close()
        st.session_state.recording = False
        st.session_state.recording_active = False
    
    total_saved = st.session_state.collected_this_session
    status_ph.success(
        f"✅ Selesai! **{total_saved}** sampel disimpan ke `{DATASET_FILES[exercise]}` "
        f"(label: **{label}**)"
    )
    
    refresh_data()


# ── Preview dataset (selalu tampil) ───────────────────────────────────────────
st.divider()
st.markdown("### 📂 Preview Dataset")

tab_curl, tab_squat, tab_pushup = st.tabs(["💪 Curl Dataset", "🏋️ Squat Dataset", "🤸 Pushup Dataset"])

# Tab Curl
with tab_curl:
    df_show = load_dataset("curl")
    if df_show.empty:
        st.info("Belum ada data untuk Curl. Pilih Curl dari dropdown dan mulai rekam!")
    else:
        col_names = ["left_elbow", "right_elbow", "symmetry", "shoulder_diff", "wrist_height", "torso_lean", "label"]
        if len(df_show.columns) == len(col_names):
            df_show.columns = col_names
        
        counts_ex = df_show["label"].value_counts() if "label" in df_show.columns else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Curl Data", len(df_show))
        c2.metric("🔴 Bad", counts_ex.get("bad", 0))
        c3.metric("🟡 Good", counts_ex.get("good", 0))
        c4.metric("🟢 Perfect", counts_ex.get("perfect", 0))
        
        with st.expander("Lihat data Curl (50 baris terakhir)"):
            st.dataframe(df_show.tail(50), use_container_width=True, hide_index=True)
        
        csv_bytes = df_show.to_csv(index=False, header=False).encode()
        st.download_button(
            label="⬇️ Download dataset_curl.csv",
            data=csv_bytes,
            file_name="dataset_curl.csv",
            mime="text/csv",
        )

# Tab Squat
with tab_squat:
    df_show = load_dataset("squat")
    if df_show.empty:
        st.info("Belum ada data untuk Squat. Pilih Squat dari dropdown dan mulai rekam!")
    else:
        col_names = ["knee_angle", "hip_angle", "knee_cave", "depth_ratio", "symmetry", "torso_lean", "label"]
        if len(df_show.columns) == len(col_names):
            df_show.columns = col_names
        
        counts_ex = df_show["label"].value_counts() if "label" in df_show.columns else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Squat Data", len(df_show))
        c2.metric("🔴 Bad", counts_ex.get("bad", 0))
        c3.metric("🟡 Good", counts_ex.get("good", 0))
        c4.metric("🟢 Perfect", counts_ex.get("perfect", 0))
        
        with st.expander("Lihat data Squat (50 baris terakhir)"):
            st.dataframe(df_show.tail(50), use_container_width=True, hide_index=True)
        
        csv_bytes = df_show.to_csv(index=False, header=False).encode()
        st.download_button(
            label="⬇️ Download dataset_squat.csv",
            data=csv_bytes,
            file_name="dataset_squat.csv",
            mime="text/csv",
        )

# Tab Pushup
with tab_pushup:
    df_show = load_dataset("pushup")
    if df_show.empty:
        st.info("Belum ada data untuk Pushup. Pilih Pushup dari dropdown dan mulai rekam!")
    else:
        col_names = ["left_elbow", "right_elbow", "body_angle", "symmetry", "hip_deviation", "hand_width", "label"]
        if len(df_show.columns) == len(col_names):
            df_show.columns = col_names
        
        counts_ex = df_show["label"].value_counts() if "label" in df_show.columns else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pushup Data", len(df_show))
        c2.metric("🔴 Bad", counts_ex.get("bad", 0))
        c3.metric("🟡 Good", counts_ex.get("good", 0))
        c4.metric("🟢 Perfect", counts_ex.get("perfect", 0))
        
        with st.expander("Lihat data Pushup (50 baris terakhir)"):
            st.dataframe(df_show.tail(50), use_container_width=True, hide_index=True)
        
        csv_bytes = df_show.to_csv(index=False, header=False).encode()
        st.download_button(
            label="⬇️ Download dataset_pushup.csv",
            data=csv_bytes,
            file_name="dataset_pushup.csv",
            mime="text/csv",
        )

st.divider()
st.caption("📁 **Dataset:** `dataset_curl.csv`, `dataset_squat.csv`, `dataset_pushup.csv`")
