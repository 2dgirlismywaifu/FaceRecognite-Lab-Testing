from config import FACE_RECOGNITION_MODEL
from .insightface_services import InsightFaceAttendanceSystem
from .video_services import VideoService
from .facenet_services import FaceNetAttendanceSystem
from flask_socketio import SocketIO
from .internal_services import init_internal_services_socketio

_face_system = None
_socketio = None

def init_app(socketio: SocketIO):
    global _socketio
    _socketio = socketio
    init_internal_services_socketio(socketio)


def init_faces():
    """Initialize the face recognition system and return the instance."""
    global _face_system
    if _face_system is None:
        _face_system = create_face_recognition_system()
    return _face_system


def create_face_recognition_system():
    """Factory function to create the appropriate face recognition system"""
    if FACE_RECOGNITION_MODEL == "insightface":
        return InsightFaceAttendanceSystem()
    else:  # Default to facenet
        return FaceNetAttendanceSystem()


def get_face_system():
    """Get the existing face system instance."""
    global _face_system
    return _face_system

def cleanup():
    """Clean up resources."""
    global _face_system
    if _face_system is not None:
        _face_system.cap.release()
        _face_system = None

