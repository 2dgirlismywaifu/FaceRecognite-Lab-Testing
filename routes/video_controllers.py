import cv2
from flask import Blueprint, render_template, Response

from services import get_face_system

# Create a blueprint
video_bp = Blueprint('video', __name__)


@video_bp.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@video_bp.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    """Generator function to yield frames for Flask."""
    face_system = get_face_system()
    while True:
        frame = face_system.get_frame()
        if frame is None:
            continue

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in byte format
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')