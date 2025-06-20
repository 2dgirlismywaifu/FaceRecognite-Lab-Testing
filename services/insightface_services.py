from collections import defaultdict
import csv
from datetime import datetime
import os
import threading
import time
from zoneinfo import ZoneInfo
import numpy as np
from pathlib import Path

import cv2
import requests
import insightface
from PIL import Image
from .internal_services import get_face_recognition_data, confirm_attendance
from .vidgear_services import VidGearService
from config import *
from logger import logging
from utils import *
from werkzeug.exceptions import Unauthorized, BadRequest

session = requests.Session()


class InsightFaceAttendanceSystem:
    def __init__(self):
        # Initialize device and models
        self.device_ctx = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
        logging.info(f"Using device context: {'GPU' if self.device_ctx >= 0 else 'CPU'}")

        # Initialize InsightFace
        providers = ['CUDAExecutionProvider'] if self.device_ctx >= 0 else ['CPUExecutionProvider']
        self.app = insightface.app.FaceAnalysis(
            providers=providers,
            root=INSIGHTFACE_PRETRAINED_MODEL_PATH,
            allowed_modules=INSIGHTFACE_ALLOWED_MODULES,
            name=INSIGHTFACE_PRETRAINED_MODEL
        )

        self.app.prepare(
            ctx_id=self.device_ctx,
            det_thresh=INSIGHTFACE_DET_THRESH,
            det_size=INSIGHTFACE_DET_SIZE
        )

        # File-based storage (NO memory storage)
        self.data_dir = Path("face_data")
        ensure_directory_exists(self.data_dir)
        self.embeddings_file = self.data_dir / FACE_EMBEDDING_INSIGHTFACE
        self.metadata_file = self.data_dir / FACE_EMBEDDING_METADATA_INSIGHTFACE

        # Tracking state (lightweight data only)
        self.log_file = ATTENDANCE_LOG
        self.last_api_call_time = {}
        self.recognition_counters = defaultdict(lambda: defaultdict(int))
        self.unregistered_faces = []
        self.unregistered_face_images = {}
        self.face_lock = threading.Lock()

        # Constants
        self.RECOGNITION_THRESHOLD = INSIGHTFACE_RECOGNITION_THRESHOLD
        self.SIMILARITY_THRESHOLD = INSIGHTFACE_SIMILARITY_THRESHOLD
        self.API_CALL_DELAY = API_CALL_DELAY

        # Initialize files and faces
        self._init_log_file()
        self._init_daily_checkins()
        self._init_embeddings_file()
        self.load_known_faces()
        self.cap = VidGearService(RTSP_URL)

    def _init_embeddings_file(self):
        """Initialize the embeddings file if it doesn't exist"""
        if not self.embeddings_file.exists():
            self._save_embeddings_to_file({})
            logging.info("Initialized empty embeddings file")

    def _save_embeddings_to_file(self, embeddings_dict, user_face_hashes=None):
        """Save all embeddings to a single file with enhanced metadata"""
        try:
            # Convert embeddings to numpy if needed
            storage_dict = {}
            for username, embedding in embeddings_dict.items():
                if isinstance(embedding, np.ndarray):
                    storage_dict[username] = embedding
                else:
                    storage_dict[username] = np.array(embedding)

            # Use utils function to save pickle
            save_dict_to_pickle(self.embeddings_file, storage_dict)

            # Enhanced metadata with face image hashes
            metadata = {
                "total_users": len(storage_dict),
                "usernames": list(storage_dict.keys()),
                "last_updated": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                "user_face_hashes": user_face_hashes or {},
                "model_type": "insightface"
            }

            # Use utils function to save JSON
            save_dict_to_json(self.metadata_file, metadata)
            logging.info(f"Saved {len(storage_dict)} embeddings to file")
        except Exception as e:
            logging.error(f"Error saving embeddings to file: {str(e)}")

    def _load_metadata(self):
        """Load metadata without loading embeddings"""
        metadata = load_dict_from_json(self.metadata_file)
        if not metadata:
            return {"total_users": 0, "usernames": [], "last_updated": None, "model_type": "insightface"}
        return metadata

    def _get_face_embedding_from_image(self, image):
        """Extract face embedding using InsightFace"""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # InsightFace expects BGR format
        faces = self.app.get(image)

        if faces:
            # Return the embedding of the first detected face
            return faces[0].embedding
        return None

    def _recognize_face_from_file(self, face_embedding):
        """Recognize face by reading embeddings directly from file"""
        if not self.embeddings_file.exists():
            return "Unknown", 0.0

        try:
            embeddings_dict = load_dict_from_pickle(self.embeddings_file)

            if not embeddings_dict:
                return "Unknown", 0.0

            max_similarity = -1
            recognized_name = "Unknown"

            # Ensure face_embedding is numpy array
            if not isinstance(face_embedding, np.ndarray):
                face_emb_np = np.array(face_embedding)
            else:
                face_emb_np = face_embedding

            # Compare with each stored embedding
            for username, stored_embedding in embeddings_dict.items():
                try:
                    if not isinstance(stored_embedding, np.ndarray):
                        continue

                    # Use utils function for similarity computation
                    similarity = compute_cosine_similarity_numpy(face_emb_np, stored_embedding)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        recognized_name = username if similarity > self.SIMILARITY_THRESHOLD else "Unknown"

                except Exception as e:
                    logging.info(f"Error comparing with {username}: {str(e)}")
                    continue

            return recognized_name, max_similarity

        except Exception as e:
            logging.error(f"Error reading embeddings file for recognition: {str(e)}")
            return "Unknown", 0.0

    def _update_single_user_embedding(self, username, new_embedding):
        """Update or add a single user's embedding in the file"""
        try:
            embeddings_dict = load_dict_from_pickle(self.embeddings_file)

            # Update/add the user's embedding
            if isinstance(new_embedding, np.ndarray):
                embeddings_dict[username] = new_embedding
            else:
                embeddings_dict[username] = np.array(new_embedding)

            # Save back to file
            self._save_embeddings_to_file(embeddings_dict)
            logging.info(f"Updated embedding for user: {username}")

        except Exception as e:
            logging.error(f"Error updating embedding for {username}: {str(e)}")

    def _remove_user_embedding(self, username):
        """Remove a user's embedding from the file"""
        try:
            if not self.embeddings_file.exists():
                return

            embeddings_dict = load_dict_from_pickle(self.embeddings_file)

            # Remove user if exists
            if username in embeddings_dict:
                del embeddings_dict[username]
                self._save_embeddings_to_file(embeddings_dict)
                logging.info(f"Removed embedding for user: {username}")

        except Exception as e:
            logging.error(f"Error removing embedding for {username}: {str(e)}")

    @property
    def known_names(self):
        """Get all known usernames (lightweight operation)"""
        metadata = self._load_metadata()
        return metadata.get("usernames", [])

    def _init_log_file(self):
        """Initialize the attendance log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as csvfile:
                csv.writer(csvfile).writerow(['Name', 'DayAttendance', 'TimeAttendance', 'Rate', 'Type'])

    def _init_daily_checkins(self):
        """Initialize daily check-ins from the attendance log file"""
        self.daily_checkins = {}
        today = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d")

        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        name = row.get('Name')
                        day = row.get('DayAttendance')
                        type_attendance = row.get('Type')

                        if day == today and name and type_attendance == 'check-in':
                            self.daily_checkins[name] = today

                logging.info(f"Loaded {len(self.daily_checkins)} daily check-ins from log file")
            except Exception as e:
                logging.error(f"Error loading daily check-ins: {str(e)}")

    def load_known_faces(self):
        """Load face embeddings from S3 URLs and save to file"""
        logging.info("Loading known faces from S3 using InsightFace...")

        face_data = get_face_recognition_data()
        if not face_data:
            return

        # Build current face data with image URLs
        current_users = {}
        for user_data in face_data:
            username = user_data.get("user_name")
            face_urls = user_data.get("face_images", [])
            if username:
                current_users[username] = sorted(face_urls)

        metadata = self._load_metadata()
        existing_users = set(metadata.get("usernames", []))

        # Check if we need to update
        needs_update = False
        users_to_update = set()

        # Check for user list changes
        current_user_set = set(current_users.keys())
        existing_user_set = set(existing_users)

        if current_user_set != existing_user_set:
            # Only new users need processing, not all users
            new_users = current_user_set - existing_user_set
            removed_users = existing_user_set - current_user_set

            logging.info(f"User list changes - New: {new_users}, Removed: {removed_users}")
            users_to_update.update(new_users)
            needs_update = True

            # Remove embeddings for deleted users
            for username in removed_users:
                self._remove_user_embedding(username)

        # Always check hash changes for ALL current users (including existing ones)
        if 'user_face_hashes' in metadata:
            stored_hashes = metadata['user_face_hashes']
            for username, face_urls in current_users.items():
                current_hash = create_s3_url_stable_hash(face_urls)
                if stored_hashes.get(username) != current_hash:
                    logging.info(f"Face images updated for user: {username}")
                    users_to_update.add(username)
                    needs_update = True
        else:
            # No stored hashes, need to process all users
            users_to_update.update(current_users.keys())
            needs_update = True

        if not needs_update:
            logging.info("Face data is up to date, skipping reload")
            return

        # Process only changed users
        embeddings_dict = load_dict_from_pickle(self.embeddings_file) if self.embeddings_file.exists() else {}
        user_face_hashes = metadata.get('user_face_hashes', {})

        for user_data in face_data:
            username = user_data.get("user_name")
            face_urls = user_data.get("face_images", [])

            # Only process users that need updates
            if username and face_urls and username in users_to_update:
                logging.info(f"Processing {len(face_urls)} images for user: {username}")
                user_embeddings = self._process_user_faces_from_urls(
                    username, face_urls)

                if user_embeddings:
                    # Average the embeddings
                    avg_embedding = np.mean(user_embeddings, axis=0)
                    embeddings_dict[username] = avg_embedding
                    user_face_hashes[username] = create_s3_url_stable_hash(
                        face_urls)

        # Save embeddings and enhanced metadata
        if embeddings_dict:
            self._save_embeddings_to_file(embeddings_dict, user_face_hashes)
            logging.info(f"Updated {len(embeddings_dict)} user embeddings")

    def _process_user_faces_from_urls(self, username, face_urls):
        """Process face images from URLs and compute embeddings using InsightFace"""
        user_embeddings = []

        for url in face_urls:
            try:
                # Download image
                img = download_image_from_url(session, url)
                if img is None:
                    continue

                # Get embedding using InsightFace
                embedding = self._get_face_embedding_from_image(img)

                if embedding is not None:
                    user_embeddings.append(embedding)
                    logging.info(f"Processed face from URL: {clean_filename_from_url(url)}")
                else:
                    logging.info(f"No face detected in image from URL: {clean_filename_from_url(url)}")

            except Exception as e:
                logging.info(f"Error processing URL: {str(e)}")

        return user_embeddings

    def update_user_embeddings(self, api_key, username, face_urls):
        """Update face embeddings for a specific user"""
        if not api_key or api_key != ATTENDANCE_DEVICE_API_KEY:
            raise Unauthorized("This resource is not allow to be accessed")
        if not username:
            raise BadRequest("Username is required")
        if not face_urls:
            raise BadRequest("Face URLs are required")

        logging.info(f"Updating face embeddings for user: {username}")

        # Process face images
        user_embeddings = self._process_user_faces_from_urls(
            username, face_urls)

        if user_embeddings:
            # Average embeddings and update in file
            avg_embedding = np.mean(user_embeddings, axis=0)
            self._update_single_user_embedding(username, avg_embedding)
            logging.info(f"Successfully updated embeddings for user: {username}")
            return True
        else:
            logging.info(f"Failed to process new face images for user: {username}")
            return False

    def add_user_embedding(self, username, user_embeddings):
        """Add or update a user's face embedding"""
        if user_embeddings:
            # Convert to numpy arrays if needed
            numpy_embeddings = []
            for emb in user_embeddings:
                if isinstance(emb, np.ndarray):
                    numpy_embeddings.append(emb)
                else:
                    numpy_embeddings.append(np.array(emb))

            # Average the embeddings
            avg_embedding = np.mean(numpy_embeddings, axis=0)
            self._update_single_user_embedding(username, avg_embedding)

    def recognize_face(self, face_embedding):
        """Identify a face by comparing with known embeddings (file-based)"""
        return self._recognize_face_from_file(face_embedding)

    def log_attendance(self, name, day_attendance, time_attendance, rate, type_attendance):
        """Record attendance in log file"""
        if name == "Unknown":
            return False

        with open(self.log_file, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow(
                [name, day_attendance, time_attendance, rate, type_attendance])

    def process_frame(self, frame):
        """Process video frame and identify faces using InsightFace"""
        annotated_frame = frame.copy()
        recognized_faces = []
        unregistered = []

        try:
            # Convert frame to BGR for InsightFace
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Assume frame is already in BGR format (OpenCV default)
                bgr_frame = frame
            else:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Enhance contrast for better face detection
            bgr_frame = cv2.convertScaleAbs(bgr_frame, alpha=1.1, beta=10)
            # Detect faces using InsightFace
            faces = self.app.get(bgr_frame)

            for face_idx, face in enumerate(faces):
                try:
                    # Get face bounding box and embedding
                    bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
                    face_embedding = face.embedding
                    face_id = f"{face_idx}"

                    # Extract face region for unknown face handling
                    face_region = bgr_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    # Recognize face
                    name, confidence = self.recognize_face(face_embedding)

                    # Draw annotations
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    rate = f"{confidence:.2f}"
                    text = f"{name} ({rate})"
                    cv2.putText(annotated_frame, text, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Handle recognized faces
                    if name != "Unknown":
                        self._handle_recognized_face(
                            name, face_id, rate, annotated_frame)
                    else:
                        self._handle_unknown_face(face_id, face_region)

                    recognized_faces.append((name, bbox, confidence))

                except Exception as e:
                    logging.info(f"Error processing face {face_idx}: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")

        return recognized_faces, unregistered, annotated_frame

    def _handle_recognized_face(self, name, face_id, rate, annotated_frame):
        """Process a recognized face for attendance tracking"""
        # Update recognition counter
        self.recognition_counters[face_id][name] += 1

        # Reset counters for other names at this position
        for other_name in list(self.recognition_counters[face_id].keys()):
            if other_name != name:
                self.recognition_counters[face_id][other_name] = 0

        # Check if we've seen this face enough times consecutively
        if self.recognition_counters[face_id][name] >= self.RECOGNITION_THRESHOLD:
            self._record_attendance(name, face_id, rate, annotated_frame)

    def _record_attendance(self, name, face_id, rate, annotated_frame):
        """Record attendance after sufficient recognition"""
        today = datetime.now(ZoneInfo(TIMEZONE))
        today_str = today.strftime("%Y-%m-%d")

        # Check cooldown period for API calls
        current_time = time.time()
        if name in self.last_api_call_time:
            time_since_last = current_time - self.last_api_call_time[name]
            if time_since_last < self.API_CALL_DELAY:
                logging.info(f"Skipping API call for {name} (cooldown period)")
                return

        # Update API call timestamp
        self.last_api_call_time[name] = current_time
        time_attendance = today.strftime("%Y-%m-%d %H:%M:%S")

        # Determine attendance type and log accordingly
        is_checkin = name not in self.daily_checkins or self.daily_checkins[name] != today_str
        attendance_type = "check-in" if is_checkin else "check-out"

        # Log attendance
        self.log_attendance(name=name, day_attendance=today_str, time_attendance=time_attendance,
                            rate=rate, type_attendance=attendance_type)

        # Determine if check-in or check-out
        if is_checkin:
            self.daily_checkins[name] = today_str
            confirm_attendance(name=name, time_attendance=time_attendance, evedence_checkin=annotated_frame)
        else:
            confirm_attendance(name=name, time_attendance=time_attendance)

        # Reset counter
        self.recognition_counters[face_id][name] = 0

    def _handle_unknown_face(self, face_id, face):
        """Process an unknown face"""
        # Reset recognition counters for this position
        self.recognition_counters[face_id].clear()

        # Track face for potential registration
        face_id = f"unknown_{face_id}"
        with self.face_lock:
            if face_id not in self.unregistered_face_images:
                self.unregistered_face_images[face_id] = face
                self.unregistered_faces.append(face_id)

    def get_frame(self):
        """Process a frame and return annotated results"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        recognized_faces, _, annotated_frame = self.process_frame(frame)
        return annotated_frame

    def get_stats(self):
        """Get system statistics"""
        metadata = self._load_metadata()
        file_stats = get_file_stats(self.embeddings_file)

        return {
            "total_users": metadata.get("total_users", 0),
            "embeddings_file_stats": file_stats,
            "last_updated": metadata.get("last_updated"),
            "model_type": "InsightFace",
            "memory_usage": "File-based (no embeddings in memory)"
        }

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        if hasattr(self, 'app'):
            try:
                del self.app
            except:
                pass
