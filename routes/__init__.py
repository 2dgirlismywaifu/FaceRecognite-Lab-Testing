from .video_controllers import video_bp
from .face_controllers import face_bp

def init_app(app):
    """Register blueprints with the Flask application."""
    app.register_blueprint(video_bp)
    app.register_blueprint(face_bp)