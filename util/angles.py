"""
angles.py — Feature Extraction & Pose Validity Check
Fitur:
- Side view dengan threshold visibility (0.5)
- Threshold lebih akurat untuk curl & squat
- Pose validity dengan toleransi side view
"""

import numpy as np


def calculate_angle(a, b, c):
    """Hitung sudut di titik b antara a-b-c"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def calculate_distance(a, b):
    """Hitung jarak Euclidean antara dua titik"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def get_side_view_arm(lm):
    """
    Pilih sisi lengan yang paling terlihat dari side view (menghadap kanan).
    Gunakan threshold visibility (0.5) untuk memilih sisi yang valid.
    """
    vis_left = lm[13].visibility   # left elbow
    vis_right = lm[14].visibility  # right elbow
    
    # Jika salah satu visibility sangat rendah, pilih yang lain
    if vis_right > 0.5 and vis_left < 0.3:
        return True  # pakai kanan
    elif vis_left > 0.5 and vis_right < 0.3:
        return False  # pakai kiri
    else:
        # Jika keduanya cukup terlihat, pilih yang lebih tinggi
        return vis_right >= vis_left


def get_side_view_leg(lm):
    """
    Pilih sisi kaki yang paling terlihat dari side view.
    Gunakan threshold visibility (0.5) untuk memilih sisi yang valid.
    """
    vis_left = lm[25].visibility   # left knee
    vis_right = lm[26].visibility  # right knee
    
    # Jika salah satu visibility sangat rendah, pilih yang lain
    if vis_right > 0.5 and vis_left < 0.3:
        return True  # pakai kanan
    elif vis_left > 0.5 and vis_right < 0.3:
        return False  # pakai kiri
    else:
        # Jika keduanya cukup terlihat, pilih yang lebih tinggi
        return vis_right >= vis_left


def extract_curl_features(lm):
    """
    Ekstrak fitur untuk dumbbell curl – SIDE VIEW (menghadap kanan).
    Fokus pada sisi yang paling terlihat dari samping.

    Zona sudut siku (THRESHOLD LONGGAR):
      180-165 = BAD  (lengan lurus)
      165-100 = GOOD (gerakan aktif)
      100-0   = PERFECT (puncak curl, kontraksi penuh)

    Return: [elbow_angle, elbow_angle, 0, shoulder_stability, wrist_height, torso_lean]
    """
    use_right = get_side_view_arm(lm)

    if use_right:
        shoulder = [lm[12].x, lm[12].y]
        elbow    = [lm[14].x, lm[14].y]
        wrist    = [lm[16].x, lm[16].y]
        hip      = [lm[24].x, lm[24].y]
    else:
        shoulder = [lm[11].x, lm[11].y]
        elbow    = [lm[13].x, lm[13].y]
        wrist    = [lm[15].x, lm[15].y]
        hip      = [lm[23].x, lm[23].y]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # Stabilitas bahu: apakah bahu terangkat saat curl
    shoulder_stability = abs(lm[11].y - lm[12].y) * 100

    # Posisi pergelangan relatif siku (positif = wrist lebih tinggi = curl bagus)
    wrist_above_elbow = (elbow[1] - wrist[1]) * 100

    # Torso tegak (x-axis shoulder vs hip)
    torso_lean = abs(shoulder[0] - hip[0]) * 100

    # Slot simetri diisi 0 (side view 1 sisi)
    return [
        elbow_angle,
        elbow_angle,
        0.0,
        shoulder_stability,
        wrist_above_elbow,
        torso_lean
    ]


def extract_pushup_features(lm):
    """
    Ekstrak fitur advanced untuk push-up.
    TIDAK DIUBAH – sudah perfect.
    Return: [left_elbow, right_elbow, hip_angle, body_alignment, symmetry, hand_width_ratio]
    """
    l_shoulder = [lm[11].x, lm[11].y]
    l_elbow    = [lm[13].x, lm[13].y]
    l_wrist    = [lm[15].x, lm[15].y]

    r_shoulder = [lm[12].x, lm[12].y]
    r_elbow    = [lm[14].x, lm[14].y]
    r_wrist    = [lm[16].x, lm[16].y]

    l_hip  = [lm[23].x, lm[23].y]
    l_knee = [lm[25].x, lm[25].y]
    r_hip  = [lm[24].x, lm[24].y]

    left_elbow_angle  = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

    body_line_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    symmetry = abs(left_elbow_angle - right_elbow_angle)

    mid_shoulder_y = (lm[11].y + lm[12].y) / 2
    mid_hip_y      = (lm[23].y + lm[24].y) / 2
    hip_deviation  = abs(mid_hip_y - mid_shoulder_y) * 100

    hand_width     = calculate_distance(l_wrist, r_wrist)
    shoulder_width = calculate_distance(l_shoulder, r_shoulder)
    hand_width_ratio = hand_width / (shoulder_width + 1e-6)

    return [
        left_elbow_angle,
        right_elbow_angle,
        body_line_angle,
        symmetry,
        hip_deviation,
        hand_width_ratio * 100
    ]


def extract_squat_features(lm):
    """
    Ekstrak fitur untuk squat – SIDE VIEW (menghadap kanan).
    Fokus pada sisi yang paling terlihat dari samping.

    Zona sudut lutut (THRESHOLD LONGGAR):
      180-165 = BAD  (berdiri tegak)
      165-105 = GOOD (fase transisi)
      105-0   = PERFECT (bawah squat penuh)

    Return: [knee_angle, hip_angle, 0, depth_ratio, 0, torso_lean]
    """
    use_right = get_side_view_leg(lm)

    if use_right:
        hip      = [lm[24].x, lm[24].y]
        knee     = [lm[26].x, lm[26].y]
        ankle    = [lm[28].x, lm[28].y]
        shoulder = [lm[12].x, lm[12].y]
    else:
        hip      = [lm[23].x, lm[23].y]
        knee     = [lm[25].x, lm[25].y]
        ankle    = [lm[27].x, lm[27].y]
        shoulder = [lm[11].x, lm[11].y]

    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle  = calculate_angle(shoulder, hip, knee)

    # Depth: seberapa dalam squat (hip vs knee, positif = hip lebih tinggi)
    depth_ratio = (hip[1] - knee[1]) * 100

    # Torso condong (x shoulder vs hip)
    torso_lean = abs(shoulder[0] - hip[0]) * 100

    return [
        knee_angle,
        hip_angle,
        0.0,          # knee_cave – tidak terukur dari samping
        depth_ratio,
        0.0,          # symmetry – side view 1 sisi
        torso_lean
    ]


def get_angle_zone(angle, exercise):
    """
    Tentukan zona kualitas berdasarkan sudut untuk curl/squat.
    THRESHOLD LONGGAR (165° untuk bad, 100-105° untuk good).

    Curl  (sudut siku):
      >=165 = bad | 100-165 = good | <100 = perfect

    Squat (sudut lutut):
      >=165 = bad | 105-165 = good | <105 = perfect

    Returns: 'bad', 'good', atau 'perfect'
    """
    if exercise == "curl":
        if angle >= 165:
            return "bad"
        elif angle >= 100:
            return "good"
        else:
            return "perfect"
    elif exercise == "squat":
        if angle >= 165:
            return "bad"
        elif angle >= 105:
            return "good"
        else:
            return "perfect"
    return "good"


def get_pose_validity(lm, exercise=None):
    """
    Cek apakah pose valid.
    Untuk curl/squat (side view), lebih toleran karena satu sisi badan tertutup.
    Returns: (is_valid, reason)
    """
    key_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    low_conf = [i for i in key_landmarks if lm[i].visibility < 0.5]

    if exercise in ("curl", "squat"):
        # Side view: toleransi lebih longgar (satu sisi bisa tersembunyi)
        if len(low_conf) > 7:
            return False, "Terlalu banyak landmark tidak terdeteksi"
        # Minimal satu bahu terlihat
        if lm[11].visibility < 0.3 and lm[12].visibility < 0.3:
            return False, "Bahu tidak terlihat — pastikan badan dari samping"
        # Minimal satu siku terlihat
        if lm[13].visibility < 0.3 and lm[14].visibility < 0.3:
            if exercise == "curl":
                return False, "Siku tidak terlihat"
        # Minimal satu lutut terlihat
        if lm[25].visibility < 0.3 and lm[26].visibility < 0.3:
            if exercise == "squat":
                return False, "Lutut tidak terlihat"
    else:
        if len(low_conf) > 4:
            return False, "Terlalu banyak landmark tidak terdeteksi"
        if lm[11].visibility < 0.3 and lm[12].visibility < 0.3:
            return False, "Bahu tidak terlihat"

    return True, "OK"