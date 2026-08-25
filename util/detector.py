"""
detector.py — Exercise Detector v2
Rep counting dengan hysteresis & zone time tracking (base-on-time scoring)
"""

import time
import collections
from collections import Counter


class RepData:
    __slots__ = ("rep_num","score","prediction","duration","min_angle","max_angle")
    def __init__(self, rep_num, score, prediction, duration, min_angle, max_angle):
        self.rep_num   = rep_num
        self.score     = score
        self.prediction= prediction
        self.duration  = duration
        self.min_angle = min_angle
        self.max_angle = max_angle


# ── Threshold per exercise ──────────────────────────────────────────────────
_THRESHOLDS = {
    "curl":   {"start": 155, "peak": 55,  "zone_bad": 160, "zone_good": 100},
    "pushup": {"start": 155, "bottom": 110,"zone_bad": 160, "zone_good": 110},
    "squat":  {"start": 160, "bottom": 105,"zone_bad": 160, "zone_good": 105},
}


class ExerciseDetector:
    DELAY        = 0.4   # detik minimum antar rep
    HIST_SIZE    = 5     # smoothing window
    ZONE_SMOOTH  = 5     # hysteresis window

    def __init__(self):
        # stage & counter
        self._stage  = {"curl": None, "pushup": None, "squat": None}
        self._count  = {"curl": 0,    "pushup": 0,    "squat": 0   }
        self._last_t = {"curl": 0.0,  "pushup": 0.0,  "squat": 0.0 }

        # per-rep timing & range
        self._rep_start = {"curl": None, "pushup": None, "squat": None}
        self._min_ang   = {"curl": 999., "pushup": 999., "squat": 999.}
        self._max_ang   = {"curl": 0.,   "pushup": 0.,   "squat": 0.  }
        self._peaked    = {"curl": False, "pushup": False,"squat": False}

        # angle smoothing
        self._ang_hist = {ex: collections.deque(maxlen=self.HIST_SIZE)
                          for ex in ("curl","pushup","squat")}

        # zone tracking
        self._zone_times   = {ex: {"bad":0.,"good":0.,"perfect":0.}
                              for ex in ("curl","pushup","squat")}
        self._zone_last_t  = {}
        self._current_zone = {}
        self._zone_hist    = {ex: [] for ex in ("curl","pushup","squat")}

        # True setelah gerakan BENERAN mulai (sudah keluar dari zona "bad" diam
        # di posisi awal). Sebelum ini True, waktu diam/istirahat TIDAK dihitung
        # ke skor supaya jeda antar-rep gak menghukum nilai.
        self._movement_started = {ex: False for ex in ("curl","pushup","squat")}

        # Snapshot zone_times & durasi milik rep yang BARU SAJA selesai, diambil
        # SEBELUM _reset_zone() mengosongkannya. Dipakai app.py untuk menghitung
        # skor final rep tsb (bukan snapshot yang sudah ke-reset ke 0).
        self._last_zone_snapshot = {ex: ({"bad":0.,"good":0.,"perfect":0.}, 0.)
                                    for ex in ("curl","pushup","squat")}

        self.rep_log: list[RepData] = []

    # ── Zone helpers ────────────────────────────────────────────────────────
    def _raw_zone(self, exercise: str, angle: float) -> str:
        t = _THRESHOLDS[exercise]
        if angle >= t["zone_bad"]:  return "bad"
        if angle >= t["zone_good"]: return "good"
        return "perfect"

    def _smooth_zone(self, exercise: str, raw: str) -> str:
        h = self._zone_hist[exercise]
        h.append(raw)
        if len(h) > self.ZONE_SMOOTH:
            h.pop(0)
        return Counter(h).most_common(1)[0][0]

    def _update_zone(self, exercise: str, angle: float, now: float) -> str:
        raw  = self._raw_zone(exercise, angle)
        zone = self._smooth_zone(exercise, raw)

        # Tandai gerakan "beneran mulai" begitu keluar dari zona bad pertama
        # kali (mis. siku mulai menekuk). Sebelum ini, orang mungkin cuma diam
        # di posisi awal (istirahat antar-rep) — waktu itu tidak boleh dihukum.
        if not self._movement_started[exercise] and zone != "bad":
            self._movement_started[exercise] = True

        last = self._zone_last_t.get(exercise)
        if last is not None and self._movement_started[exercise]:
            self._zone_times[exercise][zone] += now - last
        self._zone_last_t[exercise]  = now
        self._current_zone[exercise] = zone
        return zone

    def _reset_zone(self, exercise: str):
        self._zone_times[exercise]      = {"bad": 0., "good": 0., "perfect": 0.}
        self._zone_hist[exercise]       = []
        self._zone_last_t.pop(exercise, None)
        self._peaked[exercise]          = False
        self._movement_started[exercise] = False

    # ── Rep lifecycle ────────────────────────────────────────────────────────
    def _start_rep(self, exercise: str, angle: float, now: float):
        if self._rep_start[exercise] is None:
            self._rep_start[exercise] = now
        self._min_ang[exercise] = angle
        self._max_ang[exercise] = angle

    def _finish_rep(self, exercise: str, score: float, prediction: str, now: float):
        dur = (now - self._rep_start[exercise]) if self._rep_start[exercise] else 0.

        # Ambil snapshot zone_times rep ini SEBELUM direset, supaya app.py bisa
        # menghitung skor final dari data rep yang UTUH (bukan yang sudah 0).
        zt_snapshot = self._zone_times[exercise].copy()
        zt_total    = sum(zt_snapshot.values())
        self._last_zone_snapshot[exercise] = (zt_snapshot, zt_total)

        self._count[exercise] += 1
        self.rep_log.append(RepData(
            self._count[exercise], score, prediction, dur,
            self._min_ang[exercise], self._max_ang[exercise]
        ))
        self._rep_start[exercise] = None
        self._last_t[exercise]    = now
        self._reset_zone(exercise)

    def _track_range(self, exercise: str, angle: float):
        self._min_ang[exercise] = min(self._min_ang[exercise], angle)
        self._max_ang[exercise] = max(self._max_ang[exercise], angle)

    # ── Detect methods ───────────────────────────────────────────────────────
    def detect_curl(self, left_angle: float, right_angle: float,
                    prediction: str, score: float):
        now   = time.time()
        angle = left_angle
        self._ang_hist["curl"].append(angle)
        self._update_zone("curl", angle, now)
        self._track_range("curl", angle)

        phase = None
        t = _THRESHOLDS["curl"]

        # posisi awal (lurus)
        if angle > t["start"]:
            if self._peaked["curl"] and self._stage["curl"] == "up":
                if now - self._last_t["curl"] > self.DELAY:
                    phase = "complete"
                    self._finish_rep("curl", score, prediction, now)
            self._stage["curl"] = "down"
            self._start_rep("curl", angle, now)

        # puncak curl
        if angle < t["peak"] and self._stage["curl"] == "down":
            self._stage["curl"]   = "up"
            self._peaked["curl"]  = True
            phase = "peak"

        return self._count["curl"], phase

    def detect_pushup(self, left_angle: float, right_angle: float,
                      prediction: str, score: float):
        now   = time.time()
        angle = (left_angle + right_angle) / 2
        self._ang_hist["pushup"].append(angle)
        self._update_zone("pushup", angle, now)
        self._track_range("pushup", angle)

        phase = None
        t = _THRESHOLDS["pushup"]

        # posisi atas (lurus)
        if angle > t["start"]:
            if self._peaked["pushup"] and self._stage["pushup"] == "down":
                if now - self._last_t["pushup"] > self.DELAY:
                    phase = "complete"
                    self._finish_rep("pushup", score, prediction, now)
            self._stage["pushup"] = "up"
            self._start_rep("pushup", angle, now)

        # posisi bawah
        if angle < t["bottom"] and self._stage["pushup"] == "up":
            self._stage["pushup"]  = "down"
            self._peaked["pushup"] = True
            phase = "bottom"

        return self._count["pushup"], phase

    def detect_squat(self, knee_angle: float, prediction: str, score: float):
        now = time.time()
        self._ang_hist["squat"].append(knee_angle)
        self._update_zone("squat", knee_angle, now)
        self._track_range("squat", knee_angle)

        phase = None
        t = _THRESHOLDS["squat"]

        # posisi berdiri
        if knee_angle > t["start"]:
            if self._peaked["squat"] and self._stage["squat"] == "down":
                if now - self._last_t["squat"] > self.DELAY:
                    phase = "complete"
                    self._finish_rep("squat", score, prediction, now)
            self._stage["squat"] = "up"
            self._start_rep("squat", knee_angle, now)

        # posisi jongkok
        if knee_angle < t["bottom"] and self._stage["squat"] == "up":
            self._stage["squat"]  = "down"
            self._peaked["squat"] = True
            phase = "bottom"

        return self._count["squat"], phase

    # ── Public helpers ───────────────────────────────────────────────────────
    def get_zone_times_and_duration(self, exercise: str) -> tuple[dict, float]:
        zt    = self._zone_times[exercise].copy()
        total = sum(zt.values())
        return zt, total

    def get_last_completed_zone_times(self, exercise: str) -> tuple[dict, float]:
        """
        Snapshot zone_times & total durasi milik REP TERAKHIR yang baru selesai,
        diambil sebelum di-reset. Pakai ini (bukan get_zone_times_and_duration)
        untuk menghitung skor final saat phase == "complete", karena
        get_zone_times_and_duration akan selalu 0 tepat setelah rep selesai.
        """
        return self._last_zone_snapshot[exercise]

    def get_current_zone(self, exercise: str) -> str:
        return self._current_zone.get(exercise, "—")

    def get_velocity(self, exercise: str) -> float:
        h = self._ang_hist[exercise]
        return (h[-1] - h[-2]) if len(h) >= 2 else 0.

    def get_tempo_warning(self, exercise: str) -> str | None:
        return "⚡ Terlalu cepat!" if abs(self.get_velocity(exercise)) > 25 else None

    def get_avg_score(self) -> float:
        return sum(r.score for r in self.rep_log) / len(self.rep_log) if self.rep_log else 0.