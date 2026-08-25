import cv2
import streamlit as st


class CameraCapture:
    """
    Context manager untuk camera capture.
    Guarantees proper release of camera resources.
    
    Usage:
        with CameraCapture(0) as cap:
            ret, frame = cap.read()
            # process frame
    """
    
    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize camera manager.
        
        Args:
            camera_id: ID camera (default 0 untuk webcam internal)
            width: Desired frame width
            height: Desired frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
    
    def __enter__(self):
        """Open camera when entering context."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Tidak dapat membuka camera {self.camera_id}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        return self.cap
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release camera when exiting context."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


class CameraValidator:
    """Utility untuk validasi camera sebelum digunakan."""
    
    @staticmethod
    def check_camera_available(camera_id=0):
        """Cek apakah camera tersedia."""
        cap = cv2.VideoCapture(camera_id)
        is_available = cap.isOpened()
        cap.release()
        return is_available
    
    @staticmethod
    def get_camera_info(camera_id=0):
        """Dapatkan informasi camera."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return {"available": False}
        
        info = {
            "available": True,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": cap.get(cv2.CAP_PROP_CONTRAST)
        }
        cap.release()
        return info
    
    @staticmethod
    def get_best_camera():
        """Coba cari camera yang available."""
        for cam_id in [0, 1, 2]:
            if CameraValidator.check_camera_available(cam_id):
                return cam_id
        return None


def safe_camera_operation(func):
    """
    Decorator untuk operasi camera yang aman.
    Menangani error dan memastikan camera release.
    """
    def wrapper(*args, **kwargs):
        cap = None
        try:
            # Cek camera availability
            cam_id = CameraValidator.get_best_camera()
            if cam_id is None:
                st.error("❌ Tidak ada camera yang terdeteksi!")
                return None
            
            with CameraCapture(cam_id) as cap:
                return func(cap, *args, **kwargs)
                
        except Exception as e:
            st.error(f"❌ Error camera: {str(e)}")
            return None
    
    return wrapper