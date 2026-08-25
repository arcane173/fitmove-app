import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "fitmove_data.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT    UNIQUE NOT NULL,
            phone       TEXT    NOT NULL,
            password    TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS workout_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            exercise     TEXT    NOT NULL,
            target_reps  INTEGER NOT NULL,
            actual_reps  INTEGER NOT NULL,
            avg_score    REAL    NOT NULL,
            consistency  REAL    NOT NULL,
            good_reps    INTEGER NOT NULL,
            bad_reps     INTEGER NOT NULL,
            feedback     TEXT    NOT NULL,
            rep_details  TEXT    NOT NULL,
            created_at   TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(email: str, phone: str, password: str) -> tuple[bool, str]:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (email, phone, password, created_at) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), phone.strip(), hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        return True, "Akun berhasil dibuat!"
    except sqlite3.IntegrityError:
        return False, "Email sudah terdaftar."
    finally:
        conn.close()


def login_user(email: str, password: str):
    """Return user dict jika berhasil, None jika gagal."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?",
        (email.lower().strip(), hash_password(password))
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_workout(user_id: int, exercise: str, target_reps: int,
                 actual_reps: int, avg_score: float, consistency: float,
                 good_reps: int, bad_reps: int, feedback: str, rep_details: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO workout_history
        (user_id, exercise, target_reps, actual_reps, avg_score, consistency,
         good_reps, bad_reps, feedback, rep_details, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, exercise, target_reps, actual_reps, round(avg_score, 1),
          round(consistency, 1), good_reps, bad_reps, feedback, rep_details,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_user_history(user_id: int) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM workout_history WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_stats(user_id: int) -> dict:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM workout_history WHERE user_id = ?", (user_id,)
    ).fetchall()
    conn.close()

    if not rows:
        return {"total_sessions": 0, "total_reps": 0, "avg_score": 0, "best_score": 0}

    rows = [dict(r) for r in rows]
    total_reps = sum(r["actual_reps"] for r in rows)
    avg_score  = sum(r["avg_score"] for r in rows) / len(rows)
    best_score = max(r["avg_score"] for r in rows)

    return {
        "total_sessions": len(rows),
        "total_reps":     total_reps,
        "avg_score":      round(avg_score, 1),
        "best_score":     round(best_score, 1),
    }
