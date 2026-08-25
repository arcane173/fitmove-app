"""
calibration.py - Kalibrasi user-specific untuk meningkatkan akurasi
"""

import numpy as np


class UserCalibration:
    """
    Kalibrasi berdasarkan proporsi tubuh user.
    Meningkatkan akurasi deteksi untuk berbagai postur tubuh.
    """
    
    def __init__(self):
        self.calibrated = False
        self.standing_shoulder_y = None
        self.standing_hip_y = None
        self.standing_knee_y = None
        self.standing_ankle_y = None
        self.body_height = None
        self.arm_length = None
        self.leg_length = None
        
    def calibrate_standing(self, landmarks):
        """
        Kalibrasi posisi berdiri ideal.
        Panggil saat user berdiri tegak di awal sesi.
        
        Args:
            landmarks: MediaPipe landmarks (33 points)
        """
        # Gunakan sisi dengan visibility lebih tinggi
        use_right = landmarks[12].visibility >= landmarks[11].visibility
        
        if use_right:
            self.standing_shoulder_y = landmarks[12].y
            self.standing_hip_y = landmarks[24].y
            self.standing_knee_y = landmarks[26].y
            self.standing_ankle_y = landmarks[28].y
        else:
            self.standing_shoulder_y = landmarks[11].y
            self.standing_hip_y = landmarks[23].y
            self.standing_knee_y = landmarks[25].y
            self.standing_ankle_y = landmarks[27].y
        
        # Hitung tinggi badan (dari bahu ke ankle)
        self.body_height = abs(self.standing_shoulder_y - self.standing_ankle_y)
        
        # Panjang lengan (bahu ke pergelangan)
        if use_right:
            shoulder = np.array([landmarks[12].x, landmarks[12].y])
            wrist = np.array([landmarks[16].x, landmarks[16].y])
        else:
            shoulder = np.array([landmarks[11].x, landmarks[11].y])
            wrist = np.array([landmarks[15].x, landmarks[15].y])
        self.arm_length = np.linalg.norm(shoulder - wrist)
        
        # Panjang tungkai (pinggul ke pergelangan kaki)
        hip = np.array([landmarks[24 if use_right else 23].x, 
                        landmarks[24 if use_right else 23].y])
        ankle = np.array([landmarks[28 if use_right else 27].x,
                          landmarks[28 if use_right else 27].y])
        self.leg_length = np.linalg.norm(hip - ankle)
        
        self.calibrated = True
        return True
    
    def get_normalized_depth(self, current_hip_y):
        """
        Normalisasi kedalaman squat berdasarkan tinggi user.
        
        Returns:
            float: 0 = berdiri, 1 = squat penuh
        """
        if not self.calibrated or self.body_height is None:
            return 0
        
        depth = (current_hip_y - self.standing_hip_y) / self.body_height
        return max(0, min(1, depth))
    
    def get_angle_adjustment(self, raw_angle, exercise):
        """
        Sesuaikan angle threshold berdasarkan proporsi tubuh.
        
        Args:
            raw_angle: Sudut yang dihitung dari landmarks
            exercise: 'curl', 'pushup', atau 'squat'
        
        Returns:
            float: Adjusted angle
        """
        if not self.calibrated:
            return raw_angle
        
        # Orang dengan lengan panjang butuh sudut lebih besar untuk curl penuh
        if exercise == 'curl' and self.arm_length is not None:
            # Adjustment: lengan panjang → kurangi threshold
            arm_factor = self.arm_length / 0.3  # Normalisasi ke panjang rata-rata
            adjusted = raw_angle * (1 - (arm_factor - 1) * 0.1)
            return max(0, min(180, adjusted))
        
        # Orang dengan tungkai panjang butuh sudut lebih dalam untuk squat
        if exercise == 'squat' and self.leg_length is not None:
            leg_factor = self.leg_length / 0.4
            adjusted = raw_angle * (1 - (leg_factor - 1) * 0.05)
            return max(0, min(180, adjusted))
        
        return raw_angle
    
    def get_personalized_thresholds(self, exercise):
        """
        Dapatkan threshold yang dipersonalisasi untuk user.
        
        Returns:
            dict: Threshold untuk bad/good/perfect
        """
        if not self.calibrated:
            # Default thresholds
            if exercise == 'curl':
                return {'bad': 160, 'good': 100, 'perfect': 55}
            elif exercise == 'squat':
                return {'bad': 160, 'good': 105, 'perfect': 90}
            else:
                return {'bad': 155, 'good': 110, 'perfect': 80}
        
        # Personalized thresholds
        if exercise == 'curl':
            arm_factor = self.arm_length / 0.3 if self.arm_length else 1
            return {
                'bad': 160,
                'good': 100 + (arm_factor - 1) * 10,
                'perfect': 55 + (arm_factor - 1) * 5
            }
        elif exercise == 'squat':
            leg_factor = self.leg_length / 0.4 if self.leg_length else 1
            return {
                'bad': 160,
                'good': 105 + (leg_factor - 1) * 8,
                'perfect': 90 + (leg_factor - 1) * 6
            }
        else:
            return {'bad': 155, 'good': 110, 'perfect': 80}
    
    def reset(self):
        """Reset kalibrasi."""
        self.calibrated = False
        self.standing_shoulder_y = None
        self.standing_hip_y = None
        self.standing_knee_y = None
        self.standing_ankle_y = None
        self.body_height = None
        self.arm_length = None
        self.leg_length = None