import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import collections
import time
import json
import pandas as pd
import os
from datetime import datetime
from collections import Counter

from database import (init_db, register_user, login_user,
                      save_workout, get_user_history, get_user_stats)
from util.angles import (extract_curl_features, extract_pushup_features,
                         extract_squat_features, get_pose_validity)
from util.detector import ExerciseDetector
from util.feedback import FeedbackSystem

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="FitMove — AI Workout Coach",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

init_db()

# ==================== MEDIAPIPE INIT (Kompatibel Semua Versi) ==================
# Coba import dengan cara yang benar untuk MediaPipe versi terbaru (1.0.0+)
try:
    from mediapipe.python import solutions as mp_solutions
    mp_pose = mp_solutions.pose
    mp_drawing = mp_solutions.drawing_utils
# Jika gagal, coba cara lama (MediaPipe 0.10.x)
except (ImportError, AttributeError):
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
    # Jika keduanya gagal, tampilkan pesan error yang jelas
    except AttributeError:
        st.error("MediaPipe tidak terinstal dengan benar. Silakan jalankan `pip install mediapipe --upgrade`.")
        st.stop()

# ==================== SESSION STATE ====================
_DEFAULTS = {
    "user":             None,
    "page":             "login",
    "selected_exercise":None,
    "target_reps":      10,
    "last_result":      None,
    "dark_mode":        True,
    "workout_done":     False,
    "workout_result":   None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

EX = {
    "pushup": {"icon":"🤸","name":"Push-Up",   "desc":"Chest · Shoulders · Triceps","color":"#00ff88","muscles":["Chest","Shoulders","Triceps"]},
    "curl":   {"icon":"💪","name":"Bicep Curl", "desc":"Biceps · Forearms",          "color":"#00d4ff","muscles":["Biceps","Forearms","Core"]},
    "squat":  {"icon":"🏋️","name":"Squat",      "desc":"Quads · Glutes · Core",      "color":"#ff6b35","muscles":["Quads","Glutes","Hamstrings"]},
}

# ==================== CSS ====================
def inject_css():
    dm = st.session_state.dark_mode
    if dm:
        bg       = "#080810"
        surface  = "rgba(255,255,255,0.04)"
        border   = "rgba(255,255,255,0.08)"
        border2  = "rgba(0,255,136,0.3)"
        text     = "#f0f0ff"
        muted    = "#6b7080"
        input_bg = "rgba(255,255,255,0.05)"
        input_bd = "rgba(255,255,255,0.1)"
        ftbg     = "#050508"
        err_bg   = "rgba(255,50,50,0.08)"
        ok_bg    = "rgba(0,255,136,0.08)"
        btn_sec  = "rgba(255,255,255,0.7)"
        pbg      = "rgba(255,255,255,0.06)"
    else:
        bg       = "#f0f2f8"
        surface  = "rgba(255,255,255,0.9)"
        border   = "rgba(0,0,0,0.07)"
        border2  = "rgba(0,180,100,0.4)"
        text     = "#0a0a1a"
        muted    = "#7080a0"
        input_bg = "rgba(255,255,255,0.8)"
        input_bd = "rgba(0,0,0,0.12)"
        ftbg     = "#e0e4f0"
        err_bg   = "rgba(220,50,50,0.08)"
        ok_bg    = "rgba(0,180,80,0.08)"
        btn_sec  = "rgba(0,0,0,0.6)"
        pbg      = "rgba(0,0,0,0.08)"

    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

#MainMenu,footer,header{{visibility:hidden}}
.block-container{{padding:0!important;max-width:100%!important}}
*{{box-sizing:border-box}}
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;-webkit-font-smoothing:antialiased}}
.stApp{{background:{bg};color:{text};min-height:100vh}}
.stApp::before{{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 60% 50% at 10% 20%,rgba(0,255,136,0.07) 0%,transparent 60%),
    radial-gradient(ellipse 50% 40% at 90% 80%,rgba(0,212,255,0.05) 0%,transparent 60%);
  animation:meshShift 20s ease-in-out infinite alternate}}
@keyframes meshShift{{0%{{transform:scale(1)}}100%{{transform:scale(1.1) rotate(2deg)}}}}
.stApp::after{{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(rgba(0,255,136,0.02) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,255,136,0.02) 1px,transparent 1px);
  background-size:60px 60px;
  mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black 30%,transparent 100%)}}
.main>div{{position:relative;z-index:1}}

.fm-logo-ring{{width:38px;height:38px;border-radius:10px;
  background:linear-gradient(135deg,#00ff88,#00d4ff);
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;
  box-shadow:0 0 20px rgba(0,255,136,0.4);animation:logoPulse 3s ease-in-out infinite}}
@keyframes logoPulse{{0%,100%{{box-shadow:0 0 20px rgba(0,255,136,0.4)}}
  50%{{box-shadow:0 0 35px rgba(0,255,136,0.7),0 0 60px rgba(0,212,255,0.2)}}}}
.fm-wordmark{{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
  letter-spacing:-.02em;background:linear-gradient(90deg,#00ff88 0%,#00d4ff 50%,#00ff88 100%);
  background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;animation:textShine 4s linear infinite}}
@keyframes textShine{{0%{{background-position:0% center}}100%{{background-position:200% center}}}}
.fm-badge{{font-size:.62rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:{muted};margin-left:4px;padding-top:6px}}

.fm-footer{{background:{ftbg};border-top:1px solid {border};padding:48px;
  margin-top:80px;text-align:center}}
.fm-footer-brand{{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
  background:linear-gradient(90deg,#00ff88,#00d4ff);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px}}
.fm-footer-sub{{font-size:.82rem;color:{muted};line-height:1.8}}
.fm-footer-pills{{display:flex;gap:8px;justify-content:center;margin:16px 0 0;flex-wrap:wrap}}
.fm-pill{{background:{surface};border:1px solid {border};border-radius:100px;
  padding:4px 14px;font-size:.72rem;color:{muted};font-family:'JetBrains Mono',monospace}}

@keyframes fadeUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}

.fm-card{{background:{surface};backdrop-filter:blur(20px) saturate(150%);
  -webkit-backdrop-filter:blur(20px) saturate(150%);border:1px solid {border};
  border-radius:20px;padding:36px;
  box-shadow:0 8px 40px rgba(0,0,0,0.2),inset 0 1px 0 rgba(255,255,255,0.05);
  position:relative;overflow:hidden}}
.fm-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,255,136,0.4),transparent)}}

.auth-icon-wrap{{width:72px;height:72px;border-radius:18px;
  background:linear-gradient(135deg,rgba(0,255,136,0.15),rgba(0,212,255,0.15));
  border:1px solid rgba(0,255,136,0.2);display:flex;align-items:center;
  justify-content:center;font-size:2rem;margin:0 auto 20px;
  box-shadow:0 0 30px rgba(0,255,136,0.15)}}
.auth-title{{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
  letter-spacing:-.02em;color:{text};text-align:center;margin-bottom:6px}}
.auth-sub{{font-size:.9rem;color:{muted};text-align:center;margin-bottom:28px}}

.stTextInput>div>div>input,.stNumberInput>div>div>input{{
  background:{input_bg}!important;border:1.5px solid {input_bd}!important;
  border-radius:12px!important;color:{text}!important;padding:12px 16px!important;
  font-size:.95rem!important;font-family:'DM Sans',sans-serif!important;transition:all .2s!important}}
.stTextInput>div>div>input:focus,.stNumberInput>div>div>input:focus{{
  border-color:#00ff88!important;
  box-shadow:0 0 0 3px rgba(0,255,136,0.15),0 0 20px rgba(0,255,136,0.08)!important;
  outline:none!important}}
.stTextInput>label,.stNumberInput>label{{color:{muted}!important;font-size:.78rem!important;
  font-weight:600!important;text-transform:uppercase!important;letter-spacing:.06em!important}}

div.stButton>button{{border-radius:12px!important;font-weight:600!important;
  font-size:.9rem!important;font-family:'DM Sans',sans-serif!important;
  transition:all .2s cubic-bezier(.34,1.56,.64,1)!important;cursor:pointer!important}}
div.stButton>button[kind="primary"]{{
  background:linear-gradient(135deg,#00ff88 0%,#00d4ff 100%)!important;
  border:none!important;color:#080810!important;padding:12px 24px!important;
  font-weight:700!important;box-shadow:0 4px 20px rgba(0,255,136,0.3)!important}}
div.stButton>button[kind="primary"]:hover{{
  transform:translateY(-2px) scale(1.01)!important;
  box-shadow:0 8px 32px rgba(0,255,136,0.45)!important}}
div.stButton>button[kind="secondary"]{{
  background:{surface}!important;border:1.5px solid {border}!important;
  color:{btn_sec}!important;backdrop-filter:blur(10px)!important}}
div.stButton>button[kind="secondary"]:hover{{
  border-color:rgba(0,255,136,0.4)!important;color:#00ff88!important;
  background:rgba(0,255,136,0.05)!important}}

.fm-err{{background:{err_bg};border:1px solid rgba(255,80,80,0.3);border-radius:12px;
  padding:14px 18px;color:#ff6b6b;font-size:.88rem;margin:12px 0;animation:shake .3s ease}}
@keyframes shake{{0%,100%{{transform:translateX(0)}}25%{{transform:translateX(-4px)}}75%{{transform:translateX(4px)}}}}
.fm-ok{{background:{ok_bg};border:1px solid rgba(0,255,136,0.3);border-radius:12px;
  padding:14px 18px;color:#00ff88;font-size:.88rem;margin:12px 0}}
.fm-divider{{display:flex;align-items:center;gap:16px;margin:24px 0;color:{muted};font-size:.8rem}}
.fm-divider::before,.fm-divider::after{{content:'';flex:1;height:1px;background:{border}}}

.stat-card{{background:{surface};backdrop-filter:blur(20px);border:1px solid {border};
  border-radius:16px;padding:22px 20px;text-align:center;position:relative;overflow:hidden;
  transition:transform .2s,box-shadow .2s}}
.stat-card:hover{{transform:translateY(-3px);box-shadow:0 12px 40px rgba(0,0,0,0.2)}}
.stat-card::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#00ff88,#00d4ff);opacity:0;transition:opacity .2s}}
.stat-card:hover::after{{opacity:1}}
.stat-num{{font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
  background:linear-gradient(135deg,#00ff88,#00d4ff);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;line-height:1}}
.stat-lbl{{font-size:.72rem;color:{muted};text-transform:uppercase;
  letter-spacing:.1em;margin-top:6px;font-weight:700}}

.ex-card{{background:{surface};backdrop-filter:blur(20px);border:1.5px solid {border};
  border-radius:24px;padding:44px 24px 36px;text-align:center;cursor:pointer;
  transition:all .25s cubic-bezier(.34,1.56,.64,1);position:relative;overflow:hidden;
  min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center}}
.ex-card::before{{content:'';position:absolute;inset:0;border-radius:24px;
  background:linear-gradient(135deg,var(--ex-color,#00ff88),transparent);opacity:0;transition:opacity .3s}}
.ex-card::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,transparent,var(--ex-color,#00ff88),transparent);
  opacity:0;transition:opacity .3s}}
.ex-card:hover{{transform:translateY(-8px) scale(1.02);border-color:var(--ex-color,#00ff88);
  box-shadow:0 24px 60px rgba(0,0,0,0.3)}}
.ex-card:hover::before{{opacity:.09}}.ex-card:hover::after{{opacity:1}}
.ex-card.selected{{border-color:var(--ex-color,#00ff88);
  box-shadow:0 0 40px color-mix(in srgb,var(--ex-color,#00ff88) 25%,transparent),
    inset 0 0 40px color-mix(in srgb,var(--ex-color,#00ff88) 6%,transparent)}}
.ex-card.selected::after{{opacity:1}}
.ex-icon-wrap{{width:110px;height:110px;border-radius:28px;margin:0 auto 20px;
  background:color-mix(in srgb,var(--ex-color,#00ff88) 12%,transparent);
  border:1.5px solid color-mix(in srgb,var(--ex-color,#00ff88) 28%,transparent);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 30px color-mix(in srgb,var(--ex-color,#00ff88) 20%,transparent);
  animation:iconFloat 3s ease-in-out infinite;transition:transform .2s,box-shadow .2s}}
.ex-card:hover .ex-icon-wrap{{transform:scale(1.08);
  box-shadow:0 0 50px color-mix(in srgb,var(--ex-color,#00ff88) 35%,transparent)}}
@keyframes iconFloat{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-6px)}}}}
.ex-icon{{font-size:4.8rem;display:block;line-height:1;
  filter:drop-shadow(0 0 16px var(--ex-color,#00ff88))}}
.ex-name{{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
  color:{text};margin-bottom:8px;letter-spacing:-.01em}}
.ex-desc{{font-size:.82rem;color:{muted};font-weight:500;letter-spacing:.02em;line-height:1.5}}
.ex-muscles{{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin-top:14px}}
.ex-muscle-tag{{background:color-mix(in srgb,var(--ex-color,#00ff88) 10%,transparent);
  border:1px solid color-mix(in srgb,var(--ex-color,#00ff88) 22%,transparent);
  color:var(--ex-color,#00ff88);border-radius:100px;padding:3px 10px;
  font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em}}

.target-box{{background:{surface};backdrop-filter:blur(20px);
  border:1.5px solid rgba(0,255,136,0.25);border-radius:20px;padding:28px 32px;
  box-shadow:0 0 40px rgba(0,255,136,0.08),inset 0 0 40px rgba(0,255,136,0.03);
  animation:fadeUp .3s ease both}}
.target-title{{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
  color:{text};margin-bottom:4px}}
.target-sub{{font-size:.85rem;color:{muted};margin-bottom:20px}}

.countdown-ring{{font-family:'Syne',sans-serif;font-size:11rem;font-weight:800;
  text-align:center;line-height:1;
  background:linear-gradient(135deg,#00ff88,#00d4ff,#00ff88);background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:countPulse .8s ease-in-out infinite alternate,textShine 2s linear infinite;
  filter:drop-shadow(0 0 40px rgba(0,255,136,0.3))}}
@keyframes countPulse{{from{{transform:scale(1)}}to{{transform:scale(1.04)}}}}
.countdown-sub{{text-align:center;font-size:1.1rem;color:{muted};letter-spacing:.12em;
  text-transform:uppercase;font-weight:600;margin-top:16px}}

.result-hero{{background:linear-gradient(135deg,rgba(0,255,136,0.08) 0%,
    rgba(0,212,255,0.05) 50%,rgba(255,107,53,0.03) 100%);
  border:1px solid rgba(0,255,136,0.15);border-radius:24px;padding:40px;
  text-align:center;position:relative;overflow:hidden;animation:fadeUp .4s ease both}}
.result-score{{font-family:'Syne',sans-serif;font-size:7rem;font-weight:800;
  background:linear-gradient(135deg,#00ff88,#00d4ff);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;
  filter:drop-shadow(0 0 40px rgba(0,255,136,0.4));line-height:1}}

.feedback-card{{background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.2);
  border-radius:16px;padding:24px;position:relative;overflow:hidden}}
.feedback-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#00d4ff,#00ff88)}}
.feedback-title{{font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;
  color:#00d4ff;text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px}}

.hist-row{{background:{surface};border:1px solid {border};border-radius:16px;
  padding:18px 22px;margin-bottom:10px;display:flex;align-items:center;gap:16px;
  transition:border-color .2s,transform .2s}}
.hist-row:hover{{border-color:rgba(0,255,136,0.2);transform:translateX(4px)}}

.profile-avatar{{width:110px;height:110px;border-radius:50%;
  background:linear-gradient(135deg,#00ff88,#00d4ff);
  display:flex;align-items:center;justify-content:center;font-size:3.2rem;
  margin:0 auto 20px;
  box-shadow:0 0 0 4px rgba(0,255,136,0.15),0 0 0 8px rgba(0,255,136,0.06),
    0 0 50px rgba(0,255,136,0.3);animation:logoPulse 3s ease-in-out infinite}}

.section-head{{font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;
  color:{text};letter-spacing:-.02em;margin:36px 0 18px;
  display:flex;align-items:center;gap:10px}}
.section-head::after{{content:'';flex:1;height:1px;
  background:linear-gradient(90deg,{border2},transparent)}}

.stTabs [data-baseweb="tab-list"]{{background:{surface};border:1px solid {border};
  border-radius:14px;padding:4px;gap:4px;backdrop-filter:blur(10px)}}
.stTabs [data-baseweb="tab"]{{background:transparent;border-radius:10px;
  color:{muted};font-weight:600;font-family:'DM Sans',sans-serif}}
.stTabs [aria-selected="true"]{{
  background:linear-gradient(135deg,#00ff88,#00d4ff)!important;color:#080810!important}}

.stProgress>div>div>div{{
  background:linear-gradient(90deg,#00ff88,#00d4ff)!important;
  box-shadow:0 0 10px rgba(0,255,136,0.4);border-radius:100px!important}}
.stProgress>div>div{{background:{pbg}!important;border-radius:100px!important}}

.sc-p{{color:#00ff88!important;font-weight:700}}
.sc-g{{color:#00d4ff!important;font-weight:700}}
.sc-o{{color:#ffaa00!important;font-weight:700}}
.sc-b{{color:#ff5555!important;font-weight:700}}

.workout-result-box{{background:{surface};backdrop-filter:blur(20px);
  border:1px solid {border2};border-radius:20px;padding:30px;margin-top:20px;
  animation:fadeUp .4s ease both}}

::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-track{{background:{bg}}}
::-webkit-scrollbar-thumb{{background:linear-gradient(135deg,#00ff88,#00d4ff);border-radius:10px}}
::-webkit-scrollbar-thumb:hover{{background:#00ff88}}
</style>""", unsafe_allow_html=True)


# ==================== HEADER & FOOTER ====================
def show_header(logged_in=False):
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:16px 0 4px 0">
      <div class="fm-logo-ring">⚡</div>
      <div class="fm-wordmark">FitMove</div>
      <div class="fm-badge">AI Workout Coach</div>
    </div>
    <hr style="margin:0 0 16px 0;border:none;border-top:1px solid rgba(255,255,255,0.06)">
    """, unsafe_allow_html=True)

    _, nav_col = st.columns([8, 1])
    with nav_col:
        dm = st.session_state.dark_mode
        if logged_in:
            if st.button("🚪", key="logout_btn", help="Logout"):
                st.session_state.user = None
                st.session_state.page = "login"
                st.rerun()


def show_footer():
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:10px;padding:48px;margin-top:80px;border-top:1px solid rgba(255,255,255,0.08);background:#050508">
      <div class="fm-footer-brand">FitMove</div>
      <div class="fm-footer-sub">AI Workout Coach<br>Powered by MediaPipe & Machine Learning</div>
      <div class="fm-footer-pills">
        <span class="fm-pill">🏋️ Push-Up</span>
        <span class="fm-pill">💪 Bicep Curl</span>
        <span class="fm-pill">🏋️ Squat</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ==================== AUTH ====================
def render_login():
    st.markdown(f"""
    <div style="max-width:460px;margin:80px auto 0;padding:20px">
      <div class="fm-card">
        <div class="auth-icon-wrap">⚡</div>
        <h2 class="auth-title">Selamat Datang!</h2>
        <p class="auth-sub">Masuk untuk mulai latihan dengan AI Coach</p>
    """, unsafe_allow_html=True)

    email = st.text_input("📧 Email", key="login_email")
    password = st.text_input("🔒 Password", type="password", key="login_password")

    if st.button("🚀 Masuk", type="primary", use_container_width=True):
        user = login_user(email, password)
        if user:
            st.session_state.user = user
            st.session_state.page = "home"
            st.rerun()
        else:
            st.markdown('<div class="fm-err">❌ Email atau password salah!</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="fm-divider">Belum punya akun?</div>
    """, unsafe_allow_html=True)

    if st.button("📝 Daftar Sekarang", use_container_width=True):
        st.session_state.page = "register"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_register():
    st.markdown(f"""
    <div style="max-width:460px;margin:80px auto 0;padding:20px">
      <div class="fm-card">
        <div class="auth-icon-wrap">📝</div>
        <h2 class="auth-title">Buat Akun</h2>
        <p class="auth-sub">Daftar untuk mulai latihan dengan AI Coach</p>
    """, unsafe_allow_html=True)

    email = st.text_input("📧 Gmail", key="reg_email")
    phone = st.text_input("📱 Nomor Telepon", key="reg_phone")
    password = st.text_input("🔒 Password", type="password", key="reg_password")
    confirm = st.text_input("🔒 Konfirmasi Password", type="password", key="reg_confirm")

    if st.button("✅ Daftar", type="primary", use_container_width=True):
        if password != confirm:
            st.markdown('<div class="fm-err">❌ Password tidak cocok!</div>', unsafe_allow_html=True)
        else:
            success, msg = register_user(email, phone, password)
            if success:
                st.markdown(f'<div class="fm-ok">✅ {msg}</div>', unsafe_allow_html=True)
                st.session_state.page = "login"
                st.rerun()
            else:
                st.markdown(f'<div class="fm-err">❌ {msg}</div>', unsafe_allow_html=True)

    if st.button("← Kembali ke Login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ==================== HOME ====================
def render_home():
    show_header(logged_in=True)
    st.markdown(f"""
    <div style="max-width:1200px;margin:0 auto;padding:40px 40px 20px">
      <div style="text-align:center;margin-bottom:40px">
        <h1 style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;letter-spacing:-.03em;margin:0;line-height:1.1">
          Pilih <span style="background:linear-gradient(135deg,#00ff88,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">Latihanmu</span>
        </h1>
        <p style="font-size:1.1rem;color:#6b7080;margin-top:12px">Mulai perjalanan fitness-mu dengan AI Coach</p>
      </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (key, ex) in enumerate(EX.items()):
        with cols[i]:
            selected = st.session_state.selected_exercise == key
            st.markdown(f"""
            <div class="ex-card {'selected' if selected else ''}" style="--ex-color:{ex['color']}">
              <div class="ex-icon-wrap"><span class="ex-icon">{ex['icon']}</span></div>
              <div class="ex-name">{ex['name']}</div>
              <div class="ex-desc">{ex['desc']}</div>
              <div class="ex-muscles">
                {"".join(f'<span class="ex-muscle-tag">{m}</span>' for m in ex['muscles'])}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Pilih {ex['name']}", key=f"select_{key}", type="primary" if selected else "secondary", use_container_width=True):
                st.session_state.selected_exercise = key
                st.session_state.page = "setup"
                st.rerun()

    show_footer()


# ==================== SETUP ====================
def render_setup():
    show_header(logged_in=True)
    ex_key = st.session_state.selected_exercise
    ex = EX[ex_key]

    st.markdown(f"""
    <div style="max-width:700px;margin:60px auto 0;padding:20px">
      <div class="target-box" style="--ex-color:{ex['color']}">
        <div class="target-title">{ex['icon']} Latihan {ex['name']}</div>
        <div class="target-sub">Atur target reps untuk sesi latihanmu</div>
    """, unsafe_allow_html=True)

    target = st.number_input("🎯 Target Reps", min_value=1, max_value=100, value=st.session_state.target_reps, step=1)

    st.markdown(f"""
      <div style="margin-top:20px;display:flex;gap:12px">
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🏋️ Mulai Latihan", type="primary", use_container_width=True):
            st.session_state.target_reps = int(target)
            st.session_state.page = "workout"
            st.rerun()
    with c2:
        if st.button("← Batal", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)


# ==================== WORKOUT ====================
def render_workout():
    show_header(logged_in=True)
    ex_key = st.session_state.selected_exercise
    ex = EX[ex_key]
    target = st.session_state.target_reps

    st.markdown(f"""
    <div style="max-width:900px;margin:0 auto;padding:30px 40px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="fm-logo-ring">{ex['icon']}</div>
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800">{ex['name']}</div>
            <div style="font-size:.8rem;color:#6b7080">Target: {target} reps</div>
          </div>
        </div>
        <div style="text-align:right">
          <div style="font-size:1.5rem;font-weight:700;color:#00ff88" id="rep-count">0 / {target}</div>
          <div style="font-size:.75rem;color:#6b7080;text-transform:uppercase;letter-spacing:.1em">Reps</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    # Countdown sebelum mulai
    if "countdown_start" not in st.session_state:
        st.session_state.countdown_start = time.time()

    time_elapsed = time.time() - st.session_state.countdown_start
    remaining = 5 - int(time_elapsed)

    if remaining > 0:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 0">
          <div class="countdown-ring">{remaining}</div>
          <div class="countdown-sub">Bersiap...</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.1)
        st.rerun()
    else:
        # Mulai workout
        render_workout_camera(ex_key, target)


def render_workout_camera(ex_key, target):
    # Load model
    model_path = f"model_{ex_key}.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model {model_path} tidak ditemukan!")
        return

    model = joblib.load(model_path)

    # Inisialisasi detector & feedback
    detector = ExerciseDetector(ex_key)
    feedback_system = FeedbackSystem(ex_key)

    # Session state untuk workout
    if "workout_reps" not in st.session_state:
        st.session_state.workout_reps = 0
        st.session_state.workout_feedback_list = []
        st.session_state.workout_scores = []
        st.session_state.workout_started = False

    # **PENTING: Gunakan st.camera_input untuk hosting cloud**
    camera_input = st.camera_input("Aktifkan Kamera")

    col1, col2 = st.columns([2, 1])
    with col1:
        if camera_input is not None:
            # Konversi ke format yang bisa diproses OpenCV
            import io
            from PIL import Image
            image = Image.open(camera_input)
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Proses dengan MediaPipe
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                results = pose.process(frame_bgr)

                # Gambar pose
                frame_bgr.flags.writeable = True
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 136), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 212, 255), thickness=2)
                    )

                # Ekstrak fitur & prediksi jika pose valid
                current_feedback = "Pose tidak terdeteksi. Pastikan seluruh tubuh terlihat."
                current_score = 0

                if results.pose_landmarks:
                    valid, msg = get_pose_validity(ex_key, results.pose_landmarks)
                    if valid:
                        if ex_key == "curl":
                            features = extract_curl_features(results.pose_landmarks)
                        elif ex_key == "pushup":
                            features = extract_pushup_features(results.pose_landmarks)
                        elif ex_key == "squat":
                            features = extract_squat_features(results.pose_landmarks)
                        else:
                            features = None

                        if features is not None:
                            pred = model.predict([features])[0]
                            class_label = ["bad", "good", "perfect"][pred]
                            current_feedback = feedback_system.get_feedback(class_label)
                            current_score = feedback_system.get_score(class_label)

                            # Update rep jika detektor mendeteksi gerakan baru
                            if detector.detect_rep(features):
                                st.session_state.workout_reps += 1
                                st.session_state.workout_feedback_list.append(current_feedback)
                                st.session_state.workout_scores.append(current_score)

                # Tampilkan frame
                st.image(frame_bgr, channels="BGR", use_container_width=True)

    with col2:
        # Display score
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:16px">
          <div class="stat-num" style="font-size:3rem">{st.session_state.workout_reps}</div>
          <div class="stat-lbl">Reps Selesai</div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        progress = min(st.session_state.workout_reps / target, 1.0)
        st.progress(progress)

        # Feedback saat ini
        st.markdown(f"""
        <div class="feedback-card" style="margin-top:16px">
          <div class="feedback-title">Feedback Saat Ini</div>
          <div style="font-size:.95rem;color:#f0f0ff;line-height:1.6">{current_feedback}</div>
        </div>
        """, unsafe_allow_html=True)

        # Tombol selesai
        if st.session_state.workout_reps >= target:
            st.success("🎉 Target tercapai!")

        if st.button("✅ Selesai Latihan", type="primary", use_container_width=True):
            # Hitung stats akhir
            avg_score = np.mean(st.session_state.workout_scores) if st.session_state.workout_scores else 0
            total_reps = st.session_state.workout_reps

            # Simpan ke database
            save_workout(
                user_id=st.session_state.user['id'],
                exercise=ex_key,
                target_reps=target,
                actual_reps=total_reps,
                avg_score=avg_score,
                consistency=min(1.0, total_reps / target),
                good_reps=sum(1 for s in st.session_state.workout_scores if s >= 80),
                bad_reps=sum(1 for s in st.session_state.workout_scores if s < 50),
                feedback=json.dumps(st.session_state.workout_feedback_list),
                rep_details=json.dumps(st.session_state.workout_scores)
            )

            st.session_state.workout_result = {
                'exercise': ex_key,
                'actual_reps': total_reps,
                'target_reps': target,
                'avg_score': avg_score,
                'completion': (total_reps / target) * 100
            }
            st.session_state.page = "result"
            st.rerun()


# ==================== RESULT ====================
def render_result():
    show_header(logged_in=True)
    result = st.session_state.workout_result
    if not result:
        st.session_state.page = "home"
        st.rerun()
        return

    ex_key = result['exercise']
    ex = EX[ex_key]

    # Reset workout session
    st.session_state.pop("workout_reps", None)
    st.session_state.pop("workout_feedback_list", None)
    st.session_state.pop("workout_scores", None)
    st.session_state.pop("countdown_start", None)

    score = int(result['avg_score'])

    st.markdown(f"""
    <div style="max-width:800px;margin:60px auto 0;padding:20px">
      <div class="result-hero">
        <div style="font-size:.9rem;color:#6b7080;text-transform:uppercase;letter-spacing:.12em;font-weight:700">
          {ex['icon']} {ex['name']} — Hasil Latihan
        </div>
        <div class="result-score">{score}</div>
        <div style="font-size:1.1rem;color:#f0f0ff;margin-bottom:12px">
          {result['actual_reps']} / {result['target_reps']} reps
        </div>
        <div style="font-size:1.2rem;font-weight:700;color:#00ff88">
          {result['completion']:.1f}% Completion
        </div>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:16px;margin-top:20px;justify-content:center">
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{result['actual_reps']}</div>
          <div class="stat-lbl">Reps Selesai</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{score}</div>
          <div class="stat-lbl">Avg Score</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{result['completion']:.0f}%</div>
          <div class="stat-lbl">Completion</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:30px">
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🏠 Ke Halaman Utama", type="primary", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with c2:
        if st.button("📋 Lihat Histori", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)


# ==================== HISTORY ====================
def render_history():
    show_header(logged_in=True)
    user_id = st.session_state.user['id']
    history = get_user_history(user_id)
    stats = get_user_stats(user_id)

    st.markdown(f"""
    <div style="max-width:1100px;margin:0 auto;padding:40px 40px 20px">
      <h2 style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;letter-spacing:-.03em;margin-bottom:24px">
        📋 Riwayat Latihan
      </h2>
    """, unsafe_allow_html=True)

    # Stats cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{stats['total_sessions']}</div>
          <div class="stat-lbl">Total Sesi</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{stats['total_reps']}</div>
          <div class="stat-lbl">Total Reps</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{stats['avg_score']}</div>
          <div class="stat-lbl">Avg Score</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num">{stats['best_score']}</div>
          <div class="stat-lbl">Best Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    if not history:
        st.info("Belum ada riwayat latihan. Mulai workout pertamamu!")
    else:
        # Tampilkan riwayat dalam bentuk list
        for h in history:
            ex_key = h['exercise']
            ex = EX.get(ex_key, {"icon": "🏋️", "name": ex_key})
            timestamp = datetime.fromisoformat(h['created_at'])
            st.markdown(f"""
            <div class="hist-row">
              <div style="font-size:2rem">{ex['icon']}</div>
              <div style="flex:1">
                <div style="font-weight:700;font-size:1.05rem">{ex['name']}</div>
                <div style="font-size:.8rem;color:#6b7080">{timestamp.strftime('%d %b %Y, %H:%M')}</div>
              </div>
              <div style="text-align:right">
                <div style="font-size:1.2rem;font-weight:800;color:#00ff88">{h['avg_score']}</div>
                <div style="font-size:.75rem;color:#6b7080">{h['actual_reps']}/{h['target_reps']} reps</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("← Kembali", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    show_footer()


# ==================== MAIN ====================
def main():
    inject_css()

    # Routing halaman
    page = st.session_state.page

    if page == "login":
        render_login()
    elif page == "register":
        render_register()
    elif page == "home":
        render_home()
    elif page == "setup":
        render_setup()
    elif page == "workout":
        render_workout()
    elif page == "result":
        render_result()
    elif page == "history":
        render_history()
    else:
        st.session_state.page = "login"
        st.rerun()


if __name__ == "__main__":
    main()
