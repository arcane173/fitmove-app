# 💪 FitMove — AI Workout Coach

Sistem deteksi & evaluasi gerakan workout berbasis MediaPipe dengan antarmuka Streamlit.

## Fitur
- 🔐 Login & Register (Gmail + nomor telepon)
- 🏋️ 3 jenis olahraga: Push-Up, Bicep Curl, Squat
- ⏱️ Countdown 5 detik sebelum mulai
- 🤖 Deteksi gerakan real-time via MediaPipe + ML
- 📊 Feedback form & skor per rep
- 📋 Histori workout & grafik progres di profil

---

## Cara Install & Jalankan

### 1. Install semua dependensi
```bash
pip install -r requirements.txt
```

> **Catatan:** Jika ada error mediapipe, coba:
> ```bash
> pip install mediapipe --upgrade
> ```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`

---

## Struktur File
```
fitmove/
├── app.py              ← Aplikasi utama Streamlit
├── database.py         ← Modul SQLite (user & histori)
├── requirements.txt    ← Daftar dependensi
├── fitmove_data.db     ← Database (otomatis dibuat)
├── model_curl.pkl      ← Model ML Bicep Curl
├── model_pushup.pkl    ← Model ML Push-Up
├── model_squat.pkl     ← Model ML Squat
└── util/
    ├── angles.py       ← Ekstraksi fitur pose
    ├── detector.py     ← Logika hitung rep
    └── feedback.py     ← Sistem feedback & skor
```

---

## Cara Pakai
1. Buka browser → `http://localhost:8501`
2. **Register** akun dengan Gmail & nomor telepon
3. **Login** dengan akun yang sudah dibuat
4. Pilih olahraga dan masukkan **target rep**
5. Klik **Mulai Workout** → tunggu countdown 5 detik
6. Lakukan gerakan di depan kamera
7. Lihat **feedback** setelah selesai
8. Cek **Profil** untuk riwayat & grafik progres

---

## Tips
- Pastikan seluruh tubuh terlihat di kamera
- Pencahayaan yang cukup membantu akurasi deteksi
- Untuk squat, pastikan kaki terlihat sampai lutut/pergelangan kaki
