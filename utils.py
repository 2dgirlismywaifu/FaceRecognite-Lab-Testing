import hashlib
import pickle
import json
import numpy as np
import torch
from urllib.parse import urlparse
from PIL import Image
from datetime import datetime
from pathlib import Path
from config import FACE_RECOGNITION_MODEL
from logger import logging


def background_task(app, function, *args, **kwargs):
    """Run a function in a background thread with proper Flask application context"""
    logging.info(
        f"Starting background task: {function.__name__} with args: {args}, kwargs: {kwargs}")
    with app.app_context():
        try:
            function(*args, **kwargs)
        except Exception as e:
            logging.error(f"Background task error: {str(e)}")


def clean_s3_url_for_hashing(url):
    """Extract stable part of S3 URL without expiring signature"""
    try:
        parsed = urlparse(url)

        # For S3 URLs, we only want the bucket + key part
        # Remove all query parameters (which contain signatures, timestamps, etc.)
        stable_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        return stable_url
    except Exception as e:
        logging.debug(f"Error cleaning S3 URL {url}: {str(e)}")
        return url  # Return original if parsing fails


def create_s3_url_stable_hash(urls_list):
    """Create a stable, deterministic hash from a list of URLs or strings"""
    if not urls_list:
        return ""

    # Clean URLs and sort for consistency
    cleaned_items = []
    for item in urls_list:
        clean_item = clean_s3_url_for_hashing(str(item))

        cleaned_items.append(clean_item)

    # Sort for deterministic ordering
    sorted_items = sorted(cleaned_items)

    # Include count and items in hash
    count = len(sorted_items)
    combined_string = f"count:{count}|" + "|".join(sorted_items)

    # Use SHA256 for reliable hashing
    return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()[:16]


def save_dict_to_pickle(file_path, data_dict):
    """Save any dictionary to a pickle file"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
        return True
    except Exception as e:
        logging.error(f"Error saving dict to pickle {file_path}: {str(e)}")
        return False


def load_dict_from_pickle(file_path):
    """Load dictionary from pickle file"""
    try:
        if Path(file_path).exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        logging.error(f"Error loading dict from pickle {file_path}: {str(e)}")
        return {}


def save_dict_to_json(file_path, data_dict):
    """Save any dictionary to a JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving dict to JSON {file_path}: {str(e)}")
        return False


def load_dict_from_json(file_path):
    """Load dictionary from JSON file"""
    try:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.error(f"Error loading dict from JSON {file_path}: {str(e)}")
        return {}


def convert_tensors_to_numpy(data_dict):
    """Convert PyTorch tensors in dictionary to numpy arrays"""
    converted_dict = {}
    for key, value in data_dict.items():
        if torch.is_tensor(value):
            converted_dict[key] = value.cpu().numpy()
        else:
            converted_dict[key] = value
    return converted_dict


def compute_cosine_similarity_numpy(array1, array2):
    """Compute cosine similarity between two numpy arrays"""
    try:
        # Ensure both are numpy arrays
        if torch.is_tensor(array1):
            array1 = array1.cpu().numpy()
        if torch.is_tensor(array2):
            array2 = array2.cpu().numpy()

        # Compute cosine similarity using numpy
        dot_product = np.dot(array1, array2)
        norm1 = np.linalg.norm(array1)
        norm2 = np.linalg.norm(array2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
    except Exception as e:
        logging.debug(f"Error computing cosine similarity: {str(e)}")
        return 0.0


def download_image_from_url(session, url):
    """Download and return PIL Image from URL"""
    try:
        response = session.get(url, stream=True)
        if response.status_code != 200:
            return None

        img = Image.open(response.raw)
        if FACE_RECOGNITION_MODEL == "facenet":
            img = img.convert('RGB')
        return img
    except Exception as e:
        logging.debug(f"Error downloading image from {url}: {str(e)}")
        return None


def get_file_stats(file_path):
    """Get file statistics for any file"""
    try:
        path_obj = Path(file_path)
        if path_obj.exists():
            stat = path_obj.stat()
            return {
                "exists": True,
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        return {"exists": False, "size": 0, "size_mb": 0, "modified": None}
    except Exception as e:
        logging.error(f"Error getting file stats for {file_path}: {str(e)}")
        return {"exists": False, "size": 0, "size_mb": 0, "modified": None}


def ensure_directory_exists(dir_path):
    """Ensure a directory exists, create if it doesn't"""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error creating directory {dir_path}: {str(e)}")
        return False


def clean_filename_from_url(url):
    """Extract clean filename from URL"""
    try:
        # For S3 URLs, extract just the key (filename) part
        parsed = urlparse(url)
        return parsed.path.split('/')[-1] or "unknown_file"
    except:
        return "unknown_file"
