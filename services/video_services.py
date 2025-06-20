import cv2
import threading


class VideoService:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)  # Enable hardware acceleration
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Increase FPS if camera supports it
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG for speed
        
        # Thread variables
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    
    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed = grabbed
                self.frame = frame
    
    def read(self):
        return self.grabbed, self.frame
    
    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.cap.release()