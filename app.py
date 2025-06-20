from sio import sio_handler
import os

# if os is windows:
if os.name == 'nt':
    from config import GSTREAMER_PATH
    os.add_dll_directory(GSTREAMER_PATH)

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import routes
import services
from exception.exception_handler import init_excepetion_handler
from logger import logging

socketio = SocketIO(cors_allowed_origins="*", async_mode='threading', ping_timeout=60)

def create_app():
    app = Flask(__name__, static_folder='templates', template_folder='templates')
    CORS(app)

    socketio.init_app(app)
    routes.init_app(app)
    services.init_faces()
    init_excepetion_handler(app)
    sio_handler.sio_event_handler(socketio)
    services.init_app(socketio)

    return app

app = create_app()

if __name__ == "__main__":
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logging.info(f"Error starting application: {str(e)}")
    finally:
        services.cleanup()
