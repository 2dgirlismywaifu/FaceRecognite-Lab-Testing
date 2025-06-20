from collections import defaultdict
import csv
from datetime import datetime
import os
import threading
import time
import pickle
import numpy as np
import json
from pathlib import Path

import cv2
import requests
import torch
from PIL import Image
from .internal_services import get_face_recognition_data, confirm_attendance
from .vidgear_services import VidGearService
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import *
from logger import logging
from utils import *

from werkzeug.exceptions import Unauthorized, BadRequest

session = requests.Session()


class FaceNetAttendanceSystem:
    def __init__(self):
        # Initialize device and models
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )

        model_cache_dir = Path(FACENET_PRETRAINED_MODEL_PATH)
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        # Set torch hub cache directory
        os.environ['TORCH_HOME'] = str(model_cache_dir)
        self.resnet = InceptionResnetV1(pretrained=FACENET_PRETRAINED_MODEL).eval().to(self.device)

        # File-based storage (NO memory storage)
        self.data_dir = Path("face_data")
        ensure_directory_exists(self.data_dir)  # Use utils function
        self.embeddings_file = self.data_dir / FACE_EMBEDDING_FACENET
        self.metadata_file = self.data_dir / FACE_EMBEDDING_METADATA_FACENET

        # Tracking state (lightweight data only)
        self.log_file = ATTENDANCE_LOG
        self.last_api_call_time = {}
        self.recognition_counters = defaultdict(lambda: defaultdict(int))
        self.unregistered_faces = []
        self.unregistered_face_images = {}
        self.face_lock = threading.Lock()

        # Constants
        self.RECOGNITION_THRESHOLD = FACENET_RECOGNITION_THRESHOLD
        self.SIMILARITY_THRESHOLD = FACENET_SIMILARITY_THRESHOLD
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
            # Use utils function to convert tensors
            storage_dict = convert_tensors_to_numpy(embeddings_dict)

            # Use utils function to save pickle
            save_dict_to_pickle(self.embeddings_file, storage_dict)

            # Enhanced metadata with face image hashes
            metadata = {
                "total_users": len(storage_dict),
                "usernames": list(storage_dict.keys()),
                "last_updated": datetime.now().isoformat(),
                "user_face_hashes": user_face_hashes or {}
            }

            # Use utils function to save JSON
            save_dict_to_json(self.metadata_file, metadata)
            logging.info(f"Saved {len(storage_dict)} embeddings to file")
        except Exception as e:
            logging.error(f"Error saving embeddings to file: {str(e)}")

    def _load_metadata(self):
        """Load metadata without loading embeddings"""
        # Use utils function
        metadata = load_dict_from_json(self.metadata_file)
        if not metadata:
            return {"total_users": 0, "usernames": [], "last_updated": None}
        return metadata

    def _recognize_face_from_file(self, face_embedding):
        """Recognize face by reading embeddings directly from file"""
        if not self.embeddings_file.exists():
            return "Unknown", 0.0

        try:
            # Use utils function to load embeddings
            embeddings_dict = load_dict_from_pickle(self.embeddings_file)

            if not embeddings_dict:
                return "Unknown", 0.0

            max_similarity = -1
            recognized_name = "Unknown"

            # Convert face_embedding to numpy for comparison
            if torch.is_tensor(face_embedding):
                face_emb_np = face_embedding.cpu().numpy()
            else:
                face_emb_np = face_embedding

            # Compare with each stored embedding
            for username, stored_embedding in embeddings_dict.items():
                try:
                    # Ensure stored_embedding is numpy array
                    if not isinstance(stored_embedding, np.ndarray):
                        continue

                    # Use utils function for similarity computation
                    similarity = compute_cosine_similarity_numpy(
                        face_emb_np, stored_embedding)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        recognized_name = username if similarity > self.SIMILARITY_THRESHOLD else "Unknown"

                except Exception as e:
                    logging.info(f"Error comparing with {username}: {str(e)}")
                    continue

            return recognized_name, max_similarity

        except Exception as e:
            logging.error(
                f"Error reading embeddings file for recognition: {str(e)}")
            return "Unknown", 0.0

    def _update_single_user_embedding(self, username, new_embedding):
        """Update or add a single user's embedding in the file"""
        try:
            # Use utils function to load existing embeddings
            embeddings_dict = load_dict_from_pickle(self.embeddings_file)

            # Update/add the user's embedding
            if torch.is_tensor(new_embedding):
                embeddings_dict[username] = new_embedding.cpu().numpy()
            else:
                embeddings_dict[username] = new_embedding

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

            # Use utils function to load embeddings
            embeddings_dict = load_dict_from_pickle(self.embeddings_file)

            # Remove user if exists
            if username in embeddings_dict:
                del embeddings_dict[username]
                self._save_embeddings_to_file(embeddings_dict)
                logging.info(f"Removed embedding for user: {username}")

        except Exception as e:
            logging.error(f"Error removing embedding for {username}: {str(e)}")

    @property
    def known_embeddings(self):
        """Get all known embeddings (for compatibility - not recommended)"""
        logging.warning(
            "Loading all embeddings into memory - this defeats the purpose of file-based storage")
        try:
            # Use utils function
            embeddings_dict = load_dict_from_pickle(self.embeddings_file)
            return [torch.from_numpy(emb).to(self.device) for emb in embeddings_dict.values()]
        except Exception as e:
            logging.error(f"Error loading all embeddings: {str(e)}")
            return []

    @property
    def known_names(self):
        """Get all known usernames (lightweight operation)"""
        metadata = self._load_metadata()
        return metadata.get("usernames", [])

    @property
    def embedding_dict(self):
        """Get embedding dictionary (for compatibility - not recommended)"""
        logging.warning(
            "Loading all embeddings into memory - this defeats the purpose of file-based storage")
        try:
            # Use utils function
            embeddings_dict = load_dict_from_pickle(self.embeddings_file)
            return {name: torch.from_numpy(emb).to(self.device) for name, emb in embeddings_dict.items()}
        except Exception as e:
            logging.error(f"Error loading embedding dictionary: {str(e)}")
            return {}

    def _init_log_file(self):
        """Initialize the attendance log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as csvfile:
                csv.writer(csvfile).writerow(
                    ['Name', 'DayAttendance', 'TimeAttendance', 'Rate', 'Type'])

    def _init_daily_checkins(self):
        """Initialize daily check-ins from the attendance log file"""
        self.daily_checkins = {}
        today = datetime.now().strftime("%Y-%m-%d")

        # Read the CSV file if it exists
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        name = row.get('Name')
                        day = row.get('DayAttendance')
                        type_attendance = row.get('Type')

                        # Only track the most recent status for today
                        if day == today and name and type_attendance == 'check-in':
                            self.daily_checkins[name] = today

                logging.info(f"Loaded {len(self.daily_checkins)} daily check-ins from log file")
            except Exception as e:
                logging.error(f"Error loading daily check-ins: {str(e)}")

    def load_known_faces(self):
        """Load face embeddings from S3 URLs and save to file"""
        logging.info("Loading known faces from S3...")

        face_data = get_face_recognition_data()
        if not face_data:
            return

        # Build current face data with image URLs
        current_users = {}
        for user_data in face_data:
            username = user_data.get("user_name")
            face_urls = user_data.get("face_images", [])
            if username:
                # Sort for consistent comparison
                current_users[username] = sorted(face_urls)

        metadata = self._load_metadata()
        existing_users = set(metadata.get("usernames", []))

        # Check if we need to update (improved logic)
        needs_update = False

        # Check for new or removed users
        if set(current_users.keys()) != existing_users:
            logging.info("User list has changed")
            needs_update = True

        # Check if face images have been updated for existing users
        if not needs_update and 'user_face_hashes' in metadata:
            stored_hashes = metadata['user_face_hashes']
            for username, face_urls in current_users.items():
                # Use utils function to create stable hash
                current_hash = create_s3_url_stable_hash(face_urls)
                if stored_hashes.get(username) != current_hash:
                    logging.info(f"Face images updated for user: {username}")
                    needs_update = True
                    break
        else:
            needs_update = True  # No hash data available, force update

        if not needs_update:
            logging.info("Face data is up to date, skipping reload")
            return

        # Process updates...
        embeddings_dict = {}
        user_face_hashes = {}

        for user_data in face_data:
            username = user_data.get("user_name")
            face_urls = user_data.get("face_images", [])

            if username and face_urls:
                logging.info(f"Processing {len(face_urls)} images for user: {username}")
                user_embeddings = self._process_user_faces_from_urls(username, face_urls)

                if user_embeddings:
                    avg_embedding = torch.mean(torch.stack(user_embeddings), dim=0)
                    embeddings_dict[username] = avg_embedding

                    # Use utils function to create hash
                    user_face_hashes[username] = create_s3_url_stable_hash(face_urls)

        # Save embeddings and enhanced metadata
        if embeddings_dict:
            self._save_embeddings_to_file(embeddings_dict, user_face_hashes)
            logging.info(f"Updated {len(embeddings_dict)} user embeddings")

    def _process_user_faces_from_urls(self, username, face_urls):
        """Process face images from URLs and compute embeddings (return list, don't store)"""
        user_embeddings = []

        for url in face_urls:
            try:
                # Use utils function to download image
                img = download_image_from_url(session, url)
                if img is None:
                    continue

                img_tensor = self.mtcnn(img)

                if img_tensor is not None:
                    with torch.no_grad():
                        embedding = self.resnet(img_tensor.unsqueeze(0).to(self.device))
                    user_embeddings.append(embedding[0])
                    # Use utils function for clean filename
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
            avg_embedding = torch.mean(torch.stack(user_embeddings), dim=0)
            self._update_single_user_embedding(username, avg_embedding)
            logging.info(f"Successfully updated embeddings for user: {username}")
            return True
        else:
            logging.info(f"Failed to process new face images for user: {username}")
            return False

    def add_user_embedding(self, username, user_embeddings):
        """Add or update a user's face embedding (compatibility method)"""
        if user_embeddings:
            avg_embedding = torch.mean(torch.stack(user_embeddings), dim=0)
            self._update_single_user_embedding(username, avg_embedding)

    def compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between embeddings"""
        return torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))

    def recognize_face(self, face_embedding):
        """Identify a face by comparing with known embeddings (file-based)"""
        return self._recognize_face_from_file(face_embedding)

    def log_attendance(self, name, day_attendance, time_attendance, rate, type_attendance):
        """Record attendance in log file"""
        if name == "Unknown":
            return False

        # Write the log entry use to show a temporary html page
        with open(self.log_file, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([name, day_attendance, time_attendance, rate, type_attendance])

    def process_frame(self, frame):
        """Process video frame and identify faces"""
        # Create a copy for annotation that we'll use for attendance
        annotated_frame = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Detect faces
        boxes, _ = self.mtcnn.detect(pil_image)
        recognized_faces = []
        unregistered = []

        if boxes is not None:
            for box_idx, box in enumerate(boxes):
                try:
                    box = box.astype(int)
                    face = frame_rgb[box[1]:box[3], box[0]:box[2]]
                    face_pil = Image.fromarray(face)

                    # Get embedding and identify using file-based recognition
                    face_tensor = self.mtcnn(face_pil)
                    if face_tensor is not None:
                        with torch.no_grad():
                            face_embedding = self.resnet(face_tensor.unsqueeze(0).to(self.device))

                        # Use file-based recognition
                        name, confidence = self.recognize_face(face_embedding[0])
                        face_id = f"{box_idx}"

                        # Draw annotations on annotated_frame
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                        rate = f"{confidence:.2f}"
                        text = f"{name} ({rate})"
                        cv2.putText(annotated_frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # Handle recognized faces
                        if name != "Unknown":
                            self._handle_recognized_face(name, face_id, rate, annotated_frame)
                        else:
                            self._handle_unknown_face(face_id, face)

                        recognized_faces.append((name, box, confidence))
                except Exception as e:
                    #logging.info(f"Error processing face: {str(e)}")
                    continue

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
        today = datetime.now()
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
            confirm_attendance(
                name=name, time_attendance=time_attendance, evedence_checkin=annotated_frame)
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
                face_img = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                self.unregistered_face_images[face_id] = face_img
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
        # Use utils function for file stats
        file_stats = get_file_stats(self.embeddings_file)

        return {
            "total_users": metadata.get("total_users", 0),
            "embeddings_file_stats": file_stats,
            "last_updated": metadata.get("last_updated"),
            "memory_usage": "File-based (no embeddings in memory)"
        }

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        if hasattr(self, 'mtcnn') or hasattr(self, 'resnet'):
            try:
                del self.mtcnn
                del self.resnet
                torch.cuda.empty_cache()
            except:
                pass
