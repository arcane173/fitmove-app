"""
feedback.py — Sistem Feedback & Scoring BASE ON TIME v2
Scoring = durasi di zona bad/good/perfect selama satu rep
"""

import statistics


# ── Config scoring per zona ──────────────────────────────────────────────────
_ZONE_WEIGHTS = {"bad": -10.0, "good": 5.0, "perfect": 20.0}
_BASE_SCORE   = 50.0

# ── Tips per exercise ────────────────────────────────────────────────────────
_TIPS = {
    "curl": {
        "bad":     "⬆️ Kurang dalam! Tahan sebentar di titik kontraksi (siku ditekuk penuh)",
        "good":    "👍 Bagus, terus begini!",
        "perfect": "🔥 Sempurna! Tahan sebentar, lanjutkan!",
        "peak_low":"⬆️ Angkat lebih tinggi lagi!",
        "default": "✅ Form OK",
    },
    "pushup": {
        "hip":     "⚠️ Jaga pinggul tetap lurus!",
        "body":    "⚠️ Luruskan tubuh!",
        "deep":    "⬇️ Turun lebih dalam!",
        "bad":     "⬇️ Kurang dalam! Tahan sebentar di titik kontraksi (dada dekat lantai)",
        "good":    "👍 Bagus, terus begini!",
        "perfect": "🔥 Sempurna! Lanjutkan!",
        "sym":     "⚠️ Tangan tidak seimbang!",
        "default": "✅ Form OK",
    },
    "squat": {
        "bad":     "⬇️ Kurang dalam! Tahan sebentar di titik kontraksi (posisi jongkok penuh)",
        "good":    "👍 Bagus, terus begini!",
        "perfect": "🔥 Sempurna! Lanjutkan!",
        "deep":    "⬇️ Turun lebih dalam!",
        "lean":    "⚠️ Jangan condong ke depan!",
        "default": "✅ Form OK",
    },
}

_FEEDBACK_TIPS = {
    "curl":   "💡 Tip: Jaga siku tetap di samping, curl penuh hingga <90° lalu turun perlahan.",
    "pushup": "💡 Tip: Tubuh harus lurus seperti papan dari kepala ke kaki.",
    "squat":  "💡 Tip: Turunkan hingga lutut <100° — jangan terlalu lama berdiri tegak.",
}


class FeedbackSystem:
    def __init__(self, mode: str):
        self.mode   = mode
        self.scores = []
        self.good   = 0
        self.bad    = 0

    # ── Scoring ──────────────────────────────────────────────────────────────
    @staticmethod
    def score_from_zones(zone_times: dict, total: float) -> tuple[float, str]:
        """Hitung skor dari durasi zona. Return (score, prediction)."""
        if total < 0.3:
            return 50.0, "good"

        bad_t     = zone_times.get("bad",     0.)
        good_t    = zone_times.get("good",    0.)
        perfect_t = zone_times.get("perfect", 0.)

        score = (_BASE_SCORE
                 + bad_t     * _ZONE_WEIGHTS["bad"]
                 + good_t    * _ZONE_WEIGHTS["good"]
                 + perfect_t * _ZONE_WEIGHTS["perfect"])

        # bonus & penalti
        if perfect_t >= 0.5:
            score += 10.
        if total > 0 and (bad_t / total) > 0.3:
            score -= 10.
        if perfect_t > bad_t:
            score += 5.

        score = round(max(0., min(100., score)), 1)

        if perfect_t >= 0.5 and perfect_t > bad_t:
            pred = "perfect"
        elif bad_t > perfect_t and bad_t > good_t:
            pred = "bad"
        else:
            pred = "good"

        return score, pred

    def calculate_score(self, prediction: str, features: list) -> float:
        """Legacy fallback scoring (angle-based)."""
        if self.mode == "pushup":
            return self._score_pushup_angle(features)
        return {"perfect": 80., "good": 65., "bad": 45.}.get(prediction, 50.)

    def _score_pushup_angle(self, features: list) -> float:
        left, right, body_line, symmetry, hip_dev = features[:5]
        avg   = (left + right) / 2
        score = 40.
        score += (45 if avg < 85 else 30 if avg < 100 else
                  15 if avg < 120 else 5  if avg < 140 else 0)
        if 150 < body_line < 180: score += 8
        if symmetry < 15:         score += 5
        if hip_dev  > 15:         score -= 8
        return round(max(0., min(100., score)), 1)

    # ── Update (record a completed rep) ──────────────────────────────────────
    def update(self, prediction: str, features: list):
        score = self.calculate_score(prediction, features)
        self._record(score)

    def update_with_zone_times(self, zone_times: dict, total: float, features: list):
        score, _ = self.score_from_zones(zone_times, total)
        self._record(score)
        return score

    def _record(self, score: float):
        self.scores.append(score)
        if score >= 70:
            self.good += 1
        else:
            self.bad += 1

    # ── Real-time tips ────────────────────────────────────────────────────────
    def get_realtime_tip(self, features: list, phase: str,
                         zone: str = None) -> str:
        t = _TIPS[self.mode]

        if self.mode == "curl":
            angle = features[0]
            if zone == "bad":     return t["bad"]
            if zone == "good":    return t["good"]
            if zone == "perfect": return t["perfect"]
            if phase == "peak" and angle > 65: return t["peak_low"]
            return t["default"]

        elif self.mode == "pushup":
            left, right, body_line, symmetry, hip_dev = features[:5]
            avg = (left + right) / 2
            if hip_dev   > 15:                        return t["hip"]
            if body_line < 155:                       return t["body"]
            if zone == "bad":                         return t["bad"]
            if zone == "perfect":                     return t["perfect"]
            if zone == "good":                        return t["good"]
            if avg > 120 and phase == "bottom":       return t["deep"]
            if symmetry  > 20:                        return t["sym"]
            return t["default"]

        elif self.mode == "squat":
            knee = features[0]
            if zone == "bad":                          return t["bad"]
            if zone == "good":                         return t["good"]
            if zone == "perfect":                      return t["perfect"]
            if phase == "bottom" and knee > 105:       return t["deep"]
            if len(features) > 5 and features[5] > 20:return t["lean"]
            return t["default"]

        return ""

    # ── Summary & feedback ────────────────────────────────────────────────────
    def get_percentage(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.

    def get_consistency(self) -> float:
        if len(self.scores) < 2: return 100.
        return round(max(0., 100. - statistics.stdev(self.scores) * 1.2), 1)

    def get_feedback(self) -> str:
        avg  = self.get_percentage()
        cons = self.get_consistency()
        lines = []

        if   avg >= 85: lines.append("🔥 Perfect! Form kamu sangat rapi dan konsisten.")
        elif avg >= 70: lines.append("💪 Bagus! Masih ada ruang untuk perbaikan kecil.")
        elif avg >= 55: lines.append("👍 Cukup baik. Fokus pada kontrol gerakan.")
        else:           lines.append("❌ Form masih perlu latihan. Prioritaskan teknik dulu.")

        if avg < 75:
            lines.append(_FEEDBACK_TIPS.get(self.mode, ""))

        if   cons < 65: lines.append("📊 Konsistensi perlu ditingkatkan — form kamu masih tidak stabil.")
        elif cons >= 85:lines.append("✨ Konsistensi form sangat baik!")

        return "\n".join(l for l in lines if l)

    def get_detailed_summary(self) -> dict:
        return {
            "avg_score":   round(self.get_percentage(), 1),
            "consistency": self.get_consistency(),
            "good_reps":   self.good,
            "bad_reps":    self.bad,
            "total_reps":  self.good + self.bad,
        }