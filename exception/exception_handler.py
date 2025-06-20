import traceback
from flask import Flask, json
from werkzeug.exceptions import HTTPException

from exception.custom_exception import *
from logger import logging


def init_excepetion_handler(app: Flask):

    @app.errorhandler(HTTPException)
    def handle_exception(e):
        response = e.get_response()
        response.data = json.dumps({
            "message": e.description,
        })
        logging.error(f"HTTPException: {e}")
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def handle_internal_server_error(e):
        logging.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
        # Return standardized message to the client
        response = {
            "message": "500 Internal Server Error",
        }
        return response, 500