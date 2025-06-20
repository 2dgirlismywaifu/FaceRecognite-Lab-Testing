import csv
from datetime import datetime
import json
import threading
import time
import cv2
from flask import current_app
import numpy as np
from flask_socketio import SocketIO
from PIL import Image, ImageDraw, ImageFont
import requests

from config import API_URL_BASED, ATTENDANCE_DEVICE_API_KEY
from logger import logging

session = requests.Session()
headers = {
    "Content-Type": "application/json", 
    "Accept": "application/json", 
    "X-API-Key": ATTENDANCE_DEVICE_API_KEY
}

socketio = None

def init_internal_services_socketio(sio: SocketIO):
    global socketio
    socketio = sio

def get_face_recognition_data():
    """Fetch face recognition data from API with pagination support"""
    
    try:
        # Make initial request to first page
        response = session.get(f"{API_URL_BASED}/recognite/faces-url?page=0&limit=50", headers=headers)
        if response.status_code != 200:
            return None
            
        data = json.loads(response.content.decode('utf-8'))
        if not isinstance(data, dict) or 'users' not in data:
            return None
            
        all_users = data['users']
        total_pages = data.get('total_pages', 1)
        
        # Fetch additional pages if needed
        for page in range(1, total_pages):
            response = session.get(f"{API_URL_BASED}/recognite/faces-url?page={page}&limit=50", headers=headers)
            if response.status_code == 200:
                page_data = json.loads(response.content.decode('utf-8'))
                if 'users' in page_data:
                    all_users.extend(page_data['users'])
                    
        return all_users
    except Exception as e:
        logging.info(f"Error fetching face data: {str(e)}")
        return None


def _confirm_attendance_worker(name: str, time_attendance: str, evedence_checkin=None):
    """Worker function that performs the actual attendance confirmation"""
    # Prepare request data
    form_data = {
        "user_name": name.lower(),
        "time_attendance": time_attendance
    }

    files = {}
    if evedence_checkin is not None:
        evedence_checkin_timestamp = add_timestamp_checkin_evedence(evedence_checkin, time_attendance, font_size=38)
        _, img_encoded = cv2.imencode('.jpg', evedence_checkin_timestamp)
        files = {"image": (f"{name}_{int(time.time())}.jpg", img_encoded.tobytes(), "image/jpeg")}

    logging.info(f"Sending attendance for {name}...")

    # Create headers for multipart/form-data (remove Content-Type)
    api_headers = {
        "Accept": "application/json",
        "X-API-Key": ATTENDANCE_DEVICE_API_KEY
    }

    # Make API request
    try:
        response = session.post(
            f"{API_URL_BASED}/user-attendance",
            data=form_data,
            files=files,
            headers=api_headers
        )
        logging.info(f"Response status: {response.status_code}")
        if response.status_code == 200:
            socketio.emit('evnext', {
                "name": name,
                "time": time_attendance
            })
        return response.status_code == 200
    except Exception as e:
        logging.info(f"Error confirming attendance: {str(e)}")
        return False


def confirm_attendance(name: str, time_attendance: str, evedence_checkin=None):
    """Send attendance record to database with multipart/form-data in a separate thread"""
    # Create and start a daemon thread for the API call
    thread = threading.Thread(
        target=_confirm_attendance_worker,
        args=(name, time_attendance, evedence_checkin),
        daemon=True
    )
    thread.start()
    logging.info(f"Started background thread for attendance confirmation: {name}")
    return True


def add_timestamp_checkin_evedence(image_frame, timestamp_text: str, font_size: int = 38):
    image = image_frame

    # Convert timestamp text from %Y-%m-%d %H:%M:%S to %d-%m-%Y %H:%M:%S
    dt = datetime.strptime(timestamp_text, '%Y-%m-%d %H:%M:%S')
    timestamp_text = dt.strftime('%d-%m-%Y %H:%M:%S')

    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Position at top-left corner (with padding)
    position = (45, 30)

    font_path = "font/Good_Old_DOS.ttf"  # Default font path
    font = ImageFont.truetype(font_path, font_size)
    # Create a transparent drawing layer
    draw = ImageDraw.Draw(pil_image)

    # Colors for text and outline
    text_color = (255, 255, 255)  # White text

    # Draw the main text on top
    draw.text(position, timestamp_text, font=font, fill=text_color)

    # Convert back to OpenCV format
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return result_image


def get_attendance_record(limit: int = 50) -> list:
    """Get the attendance record from the log file"""
    attendance_data = []

    # Read CSV file
    with open('attendance_log.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            attendance_data.append(row)

    # Sort by DayAttendance and TimeAttendance in descending order
    attendance_data.sort(key=lambda x: (x['DayAttendance'], x['TimeAttendance']), reverse=True)
    return attendance_data[:limit]
