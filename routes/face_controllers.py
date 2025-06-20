from flask import Blueprint, render_template, request

from services import get_face_system
from services.internal_services import get_attendance_record

face_bp = Blueprint('face_bp', __name__)

@face_bp.route('/load-face', methods=['POST'])
def reload_face():
    api_key = request.headers.get('X-API-Key')
    face_system = get_face_system()
    data = request.get_json()
    username = data.get("user")
    face_urls = data.get("images")
    result = face_system.update_user_embeddings(api_key, username, face_urls)
    if result:
        return {"message": f"Succes updated face for user {username}"}, 200
    else:
        return {"message": f"Cannot update for user {username}"}, 400


@face_bp.route('/attendance-list', methods=['GET'])
def get_attendance_list():
    result = get_attendance_record()
    return render_template('attendance-list.html', attendance_data=result)
