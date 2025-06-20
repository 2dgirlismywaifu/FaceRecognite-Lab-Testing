from flask_socketio import SocketIO
from logger import logging


def sio_event_handler(socketio: SocketIO):
    # Socket.IO event handlers
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logging.info('Client connected')

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logging.info('Client disconnected')

    @socketio.on_error_default
    def default_error_handler(e):
        return {"error": str(e)}