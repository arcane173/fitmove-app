import streamlit as st
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

# ==================== MEDIAPIPE INIT (Kompatibel) ====================
# Cara import yang kompatibel dengan MediaPipe 1.0.0+ dan 0.10.x
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

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
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("☀️" if dm else "🌙", help="Ganti Tema",
                             use_container_width=True, key="hdr_theme"):
                    st.session_state.dark_mode = not dm
                    st.rerun()
            with c2:
                if st.button("🏠", help="Beranda",
                             use_container_width=True, key="hdr_home"):
                    st.session_state.workout_done   = False
                    st.session_state.workout_result = None
                    st.session_state.page           = "home"
                    st.rerun()
            with c3:
                if st.button("👤", help="Profil",
                             use_container_width=True, key="hdr_prof"):
                    st.session_state.workout_done   = False
                    st.session_state.workout_result = None
                    st.session_state.page           = "profile"
                    st.rerun()
            with c4:
                if st.button("Logout", help="Keluar",
                             use_container_width=True, key="hdr_exit"):
                    st.session_state.user           = None
                    st.session_state.workout_done   = False
                    st.session_state.workout_result = None
                    st.session_state.page           = "login"
                    st.rerun()
        else:
            if st.button("☀️" if dm else "🌙", help="Ganti Tema",
                         use_container_width=True, key="hdr_theme_g"):
                st.session_state.dark_mode = not dm
                st.rerun()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


def show_footer():
    st.markdown("""
    <div class="fm-footer">
      <div class="fm-footer-brand">⚡ FitMove</div>
      <div class="fm-footer-sub">
        AI-Powered Workout Coach<br>
        Deteksi &amp; Evaluasi Gerakan Real-time · Berbasis MediaPipe &amp; Machine Learning
      </div>
      <div class="fm-footer-pills">
        <span class="fm-pill">MediaPipe</span>
        <span class="fm-pill">Scikit-Learn</span>
        <span class="fm-pill">Streamlit</span>
        <span class="fm-pill">OpenCV</span>
        <span class="fm-pill">SQLite</span>
      </div>
      <div style="margin-top:20px;font-size:.74rem;color:#3a3a4a">
        © 2024 FitMove · Magnum Opus Edition · Dibuat dengan ❤️ &amp; ☕
      </div>
    </div>""", unsafe_allow_html=True)


# ==================== LOGIN ====================
def page_login():
    inject_css()
    show_header(logged_in=False)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
        <div class="fm-card" style="animation:fadeUp .5s ease both">
          <div class="auth-icon-wrap">👋</div>
          <div class="auth-title">Welcome Back</div>
          <div class="auth-sub">Masuk untuk melanjutkan sesi latihan kamu</div>
        </div>""", unsafe_allow_html=True)
        email    = st.text_input("EMAIL", placeholder="kamu@gmail.com", key="li_email")
        password = st.text_input("PASSWORD", type="password",
                                 placeholder="••••••••", key="li_pw")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("MASUK →", type="primary", use_container_width=True, key="btn_li"):
            if not email or not password:
                st.markdown("<div class='fm-err'>⚠️ Isi email &amp; password dulu.</div>",
                            unsafe_allow_html=True)
            else:
                user = login_user(email, password)
                if user:
                    st.session_state.user           = user
                    st.session_state.workout_done   = False
                    st.session_state.workout_result = None
                    st.session_state.page           = "home"
                    st.rerun()
                else:
                    st.markdown("<div class='fm-err'>❌ Email atau password salah.</div>",
                                unsafe_allow_html=True)
        st.markdown("<div class='fm-divider'>atau</div>", unsafe_allow_html=True)
        if st.button("Daftar Sekarang — Gratis", type="secondary",
                     use_container_width=True, key="btn_go_reg"):
            st.session_state.page = "register"
            st.rerun()
    show_footer()


# ==================== REGISTER ====================
def page_register():
    inject_css()
    show_header(logged_in=False)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
        <div class="fm-card" style="animation:fadeUp .5s ease both">
          <div class="auth-icon-wrap">🚀</div>
          <div class="auth-title">Buat Akun</div>
          <div class="auth-sub">Mulai perjalanan fitness AI kamu hari ini — gratis selamanya</div>
        </div>""", unsafe_allow_html=True)
        email    = st.text_input("EMAIL GMAIL", placeholder="kamu@gmail.com", key="reg_email")
        phone    = st.text_input("NOMOR TELEPON", placeholder="08xx-xxxx-xxxx", key="reg_phone")
        password = st.text_input("PASSWORD", type="password",
                                 placeholder="Min. 6 karakter", key="reg_pw")
        confirm  = st.text_input("KONFIRMASI PASSWORD", type="password",
                                 placeholder="Ulangi password", key="reg_conf")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("DAFTAR SEKARANG →", type="primary",
                     use_container_width=True, key="btn_reg"):
            if not all([email, phone, password, confirm]):
                st.markdown("<div class='fm-err'>⚠️ Semua field harus diisi.</div>",
                            unsafe_allow_html=True)
            elif not email.endswith("@gmail.com"):
                st.markdown("<div class='fm-err'>⚠️ Gunakan alamat Gmail (@gmail.com).</div>",
                            unsafe_allow_html=True)
            elif len(phone) < 10:
                st.markdown("<div class='fm-err'>⚠️ Nomor telepon tidak valid.</div>",
                            unsafe_allow_html=True)
            elif len(password) < 6:
                st.markdown("<div class='fm-err'>⚠️ Password minimal 6 karakter.</div>",
                            unsafe_allow_html=True)
            elif password != confirm:
                st.markdown("<div class='fm-err'>⚠️ Password tidak cocok.</div>",
                            unsafe_allow_html=True)
            else:
                ok, msg = register_user(email, phone, password)
                if ok:
                    st.markdown(f"<div class='fm-ok'>✅ {msg} Silakan login.</div>",
                                unsafe_allow_html=True)
                    time.sleep(1.5)
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.markdown(f"<div class='fm-err'>❌ {msg}</div>",
                                unsafe_allow_html=True)
        st.markdown("<div class='fm-divider'>sudah punya akun?</div>", unsafe_allow_html=True)
        if st.button("← Kembali ke Login", type="secondary",
                     use_container_width=True, key="btn_go_li"):
            st.session_state.page = "login"
            st.rerun()
    show_footer()


# ==================== HOME ====================
def page_home():
    inject_css()
    show_header(logged_in=True)
    user  = st.session_state.user
    uname = user["email"].split("@")[0].title()
    stats = get_user_stats(user["id"])

    st.markdown(f"""
    <div style="padding:8px 0 24px 0">
      <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
           letter-spacing:-.03em;line-height:1.1">Hei, {uname} 👋</div>
      <div style="font-size:.95rem;color:#6b7080;margin-top:6px">
        Siap berkeringat hari ini? Pilih olahraga dan mulai sesi latihan kamu.
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, stats["total_sessions"], "Total Sesi"),
        (c2, stats["total_reps"],     "Total Reps"),
        (c3, f"{stats['avg_score']}%","Avg Score"),
        (c4, f"{stats['best_score']}%","Best Score"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-num'>{val}</div>
              <div class='stat-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-head'>⚡ Pilih Olahraga</div>", unsafe_allow_html=True)
    selected = st.session_state.selected_exercise

    cols = st.columns(3)
    for idx, (key, info) in enumerate(EX.items()):
        with cols[idx]:
            cls     = "ex-card selected" if selected == key else "ex-card"
            muscles = "".join([f"<span class='ex-muscle-tag'>{m}</span>"
                               for m in info["muscles"]])
            st.markdown(f"""
            <div class="{cls}" style="--ex-color:{info['color']}">
              <div class="ex-icon-wrap">
                <span class="ex-icon">{info['icon']}</span>
              </div>
              <div class="ex-name">{info['name']}</div>
              <div class="ex-desc">{info['desc']}</div>
              <div class="ex-muscles">{muscles}</div>
            </div>
            <div style="height:10px"></div>""", unsafe_allow_html=True)
            if st.button(f"Pilih {info['name']}", key=f"pick_{key}",
                         use_container_width=True):
                st.session_state.selected_exercise = key
                st.session_state.workout_done      = False
                st.session_state.workout_result    = None
                st.rerun()

    if selected:
        info = EX[selected]
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="target-box">
          <div class="target-title">{info['icon']} {info['name']} — Set Target</div>
          <div class="target-sub">Berapa repetisi yang ingin kamu selesaikan hari ini?</div>
        </div>""", unsafe_allow_html=True)
        target = st.number_input("TARGET REPETISI", min_value=1, max_value=200,
                                 value=10, step=1, key="target_input")
        ca, cb = st.columns([3, 1])
        with ca:
            if st.button("⚡ MULAI WORKOUT", type="primary",
                         use_container_width=True, key="btn_start"):
                st.session_state.target_reps   = target
                st.session_state.workout_done  = False
                st.session_state.workout_result = None
                st.session_state.page          = "countdown"
                st.rerun()
        with cb:
            if st.button("✕ Batal", type="secondary",
                         use_container_width=True, key="btn_cancel"):
                st.session_state.selected_exercise = None
                st.session_state.workout_done      = False
                st.session_state.workout_result    = None
                st.rerun()
    show_footer()


# ==================== COUNTDOWN ====================
def page_countdown():
    inject_css()
    show_header(logged_in=True)
    ex   = st.session_state.selected_exercise
    info = EX[ex]

    st.session_state.workout_done   = False
    st.session_state.workout_result = None

    st.markdown(f"""
    <div style="text-align:center;padding:20px 0 10px 0">
      <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
           text-transform:uppercase;letter-spacing:.2em;color:#6b7080;margin-bottom:4px">
        Bersiap untuk
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800">
        {info['icon']} {info['name']}
      </div>
      <div style="font-size:.9rem;color:#6b7080;margin-top:6px">
        Target: <span style="color:{info['color']};font-weight:700">
          {st.session_state.target_reps} reps
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    ph = st.empty()
    for i in range(5, 0, -1):
        with ph.container():
            st.markdown(f"<div class='countdown-ring'>{i}</div>", unsafe_allow_html=True)
            st.markdown("<div class='countdown-sub'>Bersiap…</div>", unsafe_allow_html=True)
        time.sleep(1)

    with ph.container():
        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-size:7rem;font-weight:800;
             text-align:center;background:linear-gradient(135deg,#00ff88,#00d4ff);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;filter:drop-shadow(0 0 40px rgba(0,255,136,0.5))">
          GO! 🔥
        </div>""", unsafe_allow_html=True)
    time.sleep(0.8)

    st.session_state.workout_done   = False
    st.session_state.workout_result = None
    st.session_state.page           = "workout"
    st.rerun()


# ==================== WORKOUT ====================
def page_workout():
    # Import cv2 DI DALAM fungsi (menghindari error libGL.so.1 di startup)
    import cv2
    
    inject_css()
    show_header(logged_in=True)
    ex     = st.session_state.selected_exercise
    info   = EX[ex]
    target = st.session_state.target_reps

    if (st.session_state.get("workout_done") is True
            and st.session_state.get("workout_result") is not None
            and st.session_state.get("page") == "workout"):
        show_workout_result(st.session_state.workout_result)
        return

    st.session_state.workout_done   = False
    st.session_state.workout_result = None

    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
         letter-spacing:-.02em;margin-bottom:20px">
      {info['icon']} {info['name']}
      <span style="font-size:.9rem;color:#6b7080;font-weight:400;margin-left:8px">
        Target: {target} reps
      </span>
    </div>""", unsafe_allow_html=True)

    model_map = {
        "curl":   "model_curl.pkl",
        "pushup": "model_pushup.pkl",
        "squat":  "model_squat.pkl",
    }
    extract_map = {
        "curl":   extract_curl_features,
        "pushup": extract_pushup_features,
        "squat":  extract_squat_features,
    }

    if not os.path.exists(model_map[ex]):
        st.error(f"❌ Model file '{model_map[ex]}' tidak ditemukan!")
        if st.button("← Kembali", type="secondary"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()

    try:
        model   = joblib.load(model_map[ex])
        extract = extract_map[ex]
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        if st.button("← Kembali", type="secondary"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()

    cap_test = cv2.VideoCapture(0)
    cam_ok   = cap_test.isOpened()
    cap_test.release()
    if not cam_ok:
        st.error("❌ Kamera tidak terdeteksi!")
        if st.button("← Kembali", type="secondary"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()

    col_cam, col_panel = st.columns([3, 1])
    with col_cam:
        frame_ph = st.empty()
    with col_panel:
        st.markdown("<div class='fm-card' style='padding:20px'>", unsafe_allow_html=True)
        rep_ph      = st.empty()
        score_ph    = st.empty()
        form_ph     = st.empty()
        tip_ph      = st.empty()
        progress_ph = st.empty()
        stop_btn    = st.button("⏹ Selesai", type="primary",
                                use_container_width=True, key="btn_stop")
        st.markdown("</div>", unsafe_allow_html=True)

    detector = ExerciseDetector()
    feedback = FeedbackSystem(ex)
    pose     = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap      = cv2.VideoCapture(0)
    pred_buf = collections.deque(maxlen=10)
    ex_color = {"pushup":(0,255,136),"curl":(0,212,255),"squat":(53,107,255)}[ex]

    count        = 0
    cur_score    = 0.
    tip          = ""
    last_pred    = "—"
    current_zone = "—"
    frame_n      = 0

    ZONE_LABEL = {"bad": "KURANG", "good": "BAGUS", "perfect": "SEMPURNA", "—": "—"}
    ZONE_COLOR_HEX = {"bad": "#ff5555", "good": "#ffaa00", "perfect": "#00ff88", "—": "#6b7080"}
    ZONE_COLOR_BGR = {"bad": (80,80,255), "good": (0,170,255), "perfect": (136,255,0), "—": (128,128,128)}

    _ui_last = {"count": None, "score": None, "zone": None, "tip": None}
    last_landmarks = None

    def draw_bar(f, x, y, w, h, val, mx, col):
        cv2.rectangle(f,(x,y),(x+w,y+h),(40,40,40),-1)
        fill = int(w * min(val,mx)/mx) if mx else 0
        if fill: cv2.rectangle(f,(x,y),(x+fill,y+h),col,-1)
        cv2.rectangle(f,(x,y),(x+w,y+h),(80,80,80),1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Gagal baca frame kamera.")
                break

            frame  = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            if fw > 640:
                frame = cv2.resize(frame,(640, int(fh*640/fw)))
                fh, fw = frame.shape[:2]

            frame_n += 1
            if frame_n % 2 == 0:
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                last_landmarks = res.pose_landmarks

            if last_landmarks:
                mp_drawing.draw_landmarks(
                    frame, last_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=ex_color,thickness=2,circle_radius=2),
                    mp_drawing.DrawingSpec(color=(200,200,200),thickness=1),
                )

            if frame_n % 2 == 0:
                if last_landmarks:
                    lm    = last_landmarks.landmark
                    valid, reason = get_pose_validity(lm)
                    if valid:
                        features = extract(lm)
                        pred_buf.append(model.predict(np.array([features]))[0])
                        pred      = Counter(pred_buf).most_common(1)[0][0]
                        last_pred = pred

                        if ex == "curl":
                            count, phase = detector.detect_curl(
                                features[0], features[1], pred, cur_score)
                        elif ex == "pushup":
                            count, phase = detector.detect_pushup(
                                features[0], features[1], pred, cur_score)
                        else:
                            count, phase = detector.detect_squat(
                                features[0], pred, cur_score)

                        if phase == "complete":
                            zt, dur = detector.get_last_completed_zone_times(ex)
                            if dur > 0.3:
                                cur_score, _ = FeedbackSystem.score_from_zones(zt, dur)
                            else:
                                cur_score = feedback.calculate_score(pred, features)
                            feedback._record(cur_score)
                            if detector.rep_log:
                                detector.rep_log[-1].score = cur_score
                        elif phase:
                            zt, dur = detector.get_zone_times_and_duration(ex)
                            if dur > 0.3:
                                cur_score, _ = FeedbackSystem.score_from_zones(zt, dur)

                        current_zone = detector.get_current_zone(ex)
                        tip = feedback.get_realtime_tip(features, phase, current_zone)

                        tw = detector.get_tempo_warning(ex)
                        if tw: tip = tw
                    else:
                        current_zone = "—"
                        cv2.putText(frame, reason,(10,fh-50),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(80,80,255),1)
                else:
                    current_zone = "—"
                    cv2.putText(frame,"Pose tidak terdeteksi",(10,40),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,100,100),1)

            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(220,80),(0,0,0),-1)
            cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
            cv2.putText(frame,f"{count}/{target}",(10,32),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,ex_color,2)
            draw_bar(frame,10,40,200,6,count,target,ex_color)
            cv2.putText(frame,f"{int(cur_score)}%",(10,68),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)

            zone_bgr = ZONE_COLOR_BGR.get(current_zone, (128,128,128))
            zone_txt = ZONE_LABEL.get(current_zone, "—")
            (tw_px, th_px), _ = cv2.getTextSize(zone_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            badge_x = fw - tw_px - 24
            cv2.rectangle(frame,(badge_x-10,8),(fw-8,8+th_px+16),(0,0,0),-1)
            cv2.rectangle(frame,(badge_x-10,8),(fw-8,8+th_px+16),zone_bgr,2)
            cv2.putText(frame,zone_txt,(badge_x,8+th_px+8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,zone_bgr,2)

            tc = zone_bgr
            cv2.rectangle(frame,(0,fh-32),(fw,fh),(0,0,0),-1)
            cv2.putText(frame,tip[:45],(10,fh-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,tc,1)

            frame_ph.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

            score_i = int(cur_score)
            if count != _ui_last["count"]:
                rep_ph.markdown(f"""
                <div style="text-align:center;padding:12px 0 4px">
                  <span style="font-family:'Syne',sans-serif;font-size:4.5rem;font-weight:800;
                    background:linear-gradient(135deg,#00ff88,#00d4ff);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text">{count}</span>
                  <span style="color:#6b7080;font-size:1.2rem"> / {target}</span>
                </div>""", unsafe_allow_html=True)
                progress_ph.progress(min(count/target,1.) if target else 0.)
                _ui_last["count"] = count

            if score_i != _ui_last["score"]:
                sc_col = "#00ff88" if score_i>=80 else "#ffaa00" if score_i>=60 else "#ff5555"
                score_ph.markdown(f"""
                <div style="text-align:center;font-family:'JetBrains Mono',monospace;
                  font-size:1.4rem;font-weight:600;color:{sc_col}">{score_i}%</div>""",
                unsafe_allow_html=True)
                _ui_last["score"] = score_i

            if current_zone != _ui_last["zone"]:
                zc = ZONE_COLOR_HEX.get(current_zone, "#6b7080")
                zl = ZONE_LABEL.get(current_zone, "—")
                form_ph.markdown(f"""
                <div style="text-align:center;font-size:.85rem;font-weight:700;color:{zc};
                  text-transform:uppercase;letter-spacing:.1em">● {zl}</div>""",
                unsafe_allow_html=True)
                _ui_last["zone"] = current_zone

            if tip != _ui_last["tip"]:
                tip_ph.markdown(f"""
                <div style="text-align:center;font-size:.82rem;color:#9090a0;margin:8px 0">
                  {tip}</div>""", unsafe_allow_html=True)
                _ui_last["tip"] = tip

            if count >= target or stop_btn:
                break

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        cap.release()
        pose.close()

    summary = feedback.get_detailed_summary()
    save_workout(
        user_id    = st.session_state.user["id"],
        exercise   = ex,
        target_reps= target,
        actual_reps= count,
        avg_score  = float(summary["avg_score"]),
        consistency= float(summary["consistency"]),
        good_reps  = int(summary["good_reps"]),
        bad_reps   = int(summary["bad_reps"]),
        feedback   = feedback.get_feedback(),
        rep_details= json.dumps([{
            "rep":        int(r.rep_num),
            "score":      float(round(r.score,1)),
            "prediction": str(r.prediction),
            "duration":   float(round(r.duration,2)),
        } for r in detector.rep_log]),
    )

    st.session_state.workout_result = {
        "exercise":    ex,
        "count":       count,
        "target":      target,
        "summary":     summary,
        "feedback_txt":feedback.get_feedback(),
        "rep_log":     detector.rep_log,
    }
    st.session_state.workout_done = True
    st.rerun()


# ==================== SHOW RESULT ====================
def show_workout_result(res):
    ex   = res["exercise"]
    info = EX[ex]
    s    = res["summary"]
    pct  = round(res["count"]/res["target"]*100) if res["target"] else 0
    grade = ("🏆 SEMPURNA"     if s["avg_score"]>=90 else
             "💪 BAGUS BANGET" if s["avg_score"]>=75 else
             "👍 LUMAYAN"      if s["avg_score"]>=60 else
             "💡 TERUS BERLATIH")

    st.markdown(f"""
    <div class="workout-result-box">
      <div style="text-align:center;font-family:'Syne',sans-serif;font-size:.78rem;
           font-weight:700;text-transform:uppercase;letter-spacing:.2em;
           color:#6b7080;margin-bottom:8px">
        ✅ WORKOUT SELESAI — {info['name'].upper()}
      </div>
      <div style="text-align:center;font-family:'Syne',sans-serif;font-size:6rem;
           font-weight:800;background:linear-gradient(135deg,#00ff88,#00d4ff);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text;filter:drop-shadow(0 0 40px rgba(0,255,136,0.4));line-height:1">
        {s['avg_score']}%
      </div>
      <div style="text-align:center;font-family:'Syne',sans-serif;font-size:1.5rem;
           font-weight:700;background:linear-gradient(90deg,#00ff88,#00d4ff);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text;margin:8px 0">{grade}</div>
      <div style="text-align:center;font-size:.9rem;color:#6b7080">
        {res['count']} / {res['target']} reps selesai ({pct}%)
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{s['avg_score']}%",          "Avg Score"),
        (c2, f"{s['consistency']}%",        "Konsistensi"),
        (c3, f"{s['good_reps']}/{s['total_reps']}", "Reps Bagus"),
        (c4, f"{res['count']}/{res['target']}",     "Reps Selesai"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-num'>{val}</div>
              <div class='stat-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='feedback-card'>"
                    "<div class='feedback-title'>💬 AI Coach Feedback</div>",
                    unsafe_allow_html=True)
        for line in res["feedback_txt"].split("\n"):
            if line.strip():
                st.markdown(f"<p style='margin:8px 0;font-size:.9rem;line-height:1.6'>"
                            f"{line}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_r:
        if res["rep_log"]:
            st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:.78rem;
                 font-weight:700;text-transform:uppercase;letter-spacing:.1em;
                 color:#6b7080;margin-bottom:10px">📊 Skor Per Rep</div>""",
            unsafe_allow_html=True)
            df = pd.DataFrame([{"Rep":f"#{r.rep_num}","Skor (%)":round(r.score,1)}
                                for r in res["rep_log"]])
            st.bar_chart(df.set_index("Rep")["Skor (%)"], height=200)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🏠 Beranda", type="secondary", use_container_width=True, key="res_home"):
            st.session_state.workout_done      = False
            st.session_state.workout_result    = None
            st.session_state.selected_exercise = None
            st.session_state.page              = "home"
            st.rerun()
    with col_b:
        if st.button("🔄 Ulangi", type="primary", use_container_width=True, key="res_ulang"):
            st.session_state.workout_done   = False
            st.session_state.workout_result = None
            st.session_state.page           = "countdown"
            st.rerun()
    with col_c:
        if st.button("📊 Profil", type="secondary", use_container_width=True, key="res_profil"):
            st.session_state.workout_done   = False
            st.session_state.workout_result = None
            st.session_state.page           = "profile"
            st.rerun()
    show_footer()


# ==================== PROFILE ====================
def page_profile():
    inject_css()
    show_header(logged_in=True)
    user  = st.session_state.user
    stats = get_user_stats(user["id"])
    hist  = get_user_history(user["id"])
    uname = user["email"].split("@")[0].title()

    col_info, col_stats = st.columns([1, 2.2])
    with col_info:
        st.markdown(f"""
        <div class="fm-card" style="text-align:center">
          <div class="profile-avatar">👤</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.3rem;
               font-weight:800;letter-spacing:-.02em">{uname}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
               color:#6b7080;margin-top:4px">{user['email']}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
               color:#6b7080">{user['phone']}</div>
          <div style="margin-top:16px">
            <span style="background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.2);
               color:#00ff88;border-radius:100px;padding:4px 14px;font-size:.7rem;
               font-weight:600;text-transform:uppercase;letter-spacing:.06em">MEMBER</span>
          </div>
          <div style="margin-top:12px;font-size:.72rem;color:#4a4a5a;
               font-family:'JetBrains Mono',monospace">
            Sejak {user['created_at'][:10]}
          </div>
        </div>""", unsafe_allow_html=True)

    with col_stats:
        st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:.9rem;
             font-weight:700;text-transform:uppercase;letter-spacing:.08em;
             color:#6b7080;margin-bottom:16px">📈 Statistik Keseluruhan</div>""",
        unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        for col, val, lbl in [
            (r1, stats["total_sessions"], "Total Sesi"),
            (r2, stats["total_reps"],     "Total Reps"),
        ]:
            with col:
                st.markdown(f"""
                <div class='stat-card'>
                  <div class='stat-num'>{val}</div>
                  <div class='stat-lbl'>{lbl}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        r3, r4 = st.columns(2)
        for col, val, lbl in [
            (r3, f"{stats['avg_score']}%",  "Avg Score"),
            (r4, f"{stats['best_score']}%", "Best Score"),
        ]:
            with col:
                st.markdown(f"""
                <div class='stat-card'>
                  <div class='stat-num'>{val}</div>
                  <div class='stat-lbl'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-head'>📋 Riwayat Workout</div>", unsafe_allow_html=True)

    if not hist:
        st.markdown("""
        <div class="fm-card" style="text-align:center;padding:48px">
          <div style="font-size:4rem;margin-bottom:12px;filter:grayscale(1);opacity:.4">🏋️</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
               font-weight:700;color:#6b7080">Belum ada riwayat</div>
          <div style="font-size:.85rem;color:#4a4a5a;margin-top:6px">
            Mulai latihan pertamamu sekarang! 💪
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        tab_all, tab_chart = st.tabs(["📋 Semua Riwayat", "📊 Grafik Progres"])
        with tab_all:
            for row in hist:
                ex_info  = EX.get(row["exercise"],
                                  {"icon":"🏋️","name":row["exercise"],"color":"#00ff88"})
                date_str = row["created_at"][:16].replace("T"," ")
                sc_cls   = ("sc-p" if row["avg_score"]>=90 else
                            "sc-g" if row["avg_score"]>=75 else
                            "sc-o" if row["avg_score"]>=60 else "sc-b")
                st.markdown(f"""
                <div class="hist-row">
                  <div style="width:56px;height:56px;border-radius:16px;flex-shrink:0;
                       background:color-mix(in srgb,{ex_info['color']} 12%,transparent);
                       border:1.5px solid color-mix(in srgb,{ex_info['color']} 30%,transparent);
                       display:flex;align-items:center;justify-content:center;font-size:1.9rem;
                       box-shadow:0 0 16px color-mix(in srgb,{ex_info['color']} 20%,transparent)">
                    {ex_info['icon']}
                  </div>
                  <div style="flex:1">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem">
                      {ex_info['name']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;
                         color:#6b7080;margin-top:2px">{date_str}</div>
                  </div>
                  <div style="text-align:right">
                    <div class="{sc_cls}" style="font-family:'Syne',sans-serif;
                         font-size:1.6rem;font-weight:800">{row['avg_score']}%</div>
                    <div style="font-size:.73rem;color:#6b7080">
                      {row['actual_reps']} / {row['target_reps']} reps
                    </div>
                  </div>
                  <div style="text-align:right;min-width:80px">
                    <div style="color:#00ff88;font-size:.82rem;font-weight:600">
                      ✅ {row['good_reps']} bagus</div>
                    <div style="color:#ff5555;font-size:.82rem;font-weight:600">
                      ❌ {row['bad_reps']} kurang</div>
                  </div>
                </div>""", unsafe_allow_html=True)
                with st.expander("Detail & Feedback"):
                    st.markdown(row["feedback"])
                    try:
                        reps = json.loads(row["rep_details"])
                        if reps:
                            df = pd.DataFrame(reps)
                            df.columns = ["Rep","Skor (%)","Form","Durasi (s)"]
                            st.dataframe(df, use_container_width=True, hide_index=True)
                    except Exception:
                        pass

        with tab_chart:
            if len(hist) >= 2:
                df_h = pd.DataFrame([{
                    "Tanggal": r["created_at"][:10],
                    "Skor (%)": r["avg_score"],
                    "Olahraga": EX.get(r["exercise"],{"name":r["exercise"]})["name"],
                } for r in reversed(hist)])
                st.markdown("""<div style="font-family:'Syne',sans-serif;font-weight:700;
                     margin-bottom:8px">Progres Skor</div>""", unsafe_allow_html=True)
                st.line_chart(df_h.set_index("Tanggal")["Skor (%)"], height=280)
                st.markdown("""<div style="font-family:'Syne',sans-serif;font-weight:700;
                     margin:20px 0 8px">Distribusi Olahraga</div>""", unsafe_allow_html=True)
                dist = df_h["Olahraga"].value_counts().reset_index()
                dist.columns = ["Olahraga","Sesi"]
                st.bar_chart(dist.set_index("Olahraga"), height=220)
            else:
                st.info("Butuh minimal 2 sesi untuk melihat grafik progres.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("🏠 Kembali ke Beranda", type="secondary", key="prof_home"):
        st.session_state.page = "home"
        st.rerun()
    show_footer()


# ==================== ROUTER ====================
page = st.session_state.page
user = st.session_state.user

if user is None:
    if page == "register":
        page_register()
    else:
        page_login()
else:
    if   page == "home":      page_home()
    elif page == "countdown": page_countdown()
    elif page == "workout":   page_workout()
    elif page == "profile":   page_profile()
    else:                     page_home()
