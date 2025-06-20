API_URL_BASED = "https://localhost8069/internal"
ATTENDANCE_DEVICE_API_KEY = "GUHBDUEW65768geioewewxcm"
RTSP_URL = "rtsp://admin:LKYTOI@192.168.4.22"
GSTREAMER_PATH = "D:\\builder_cache\\gstreamer\\1.0\\msvc_x86_64\\bin"
CAMERA_INDEX = 0
IS_NVIDIA_GPU = True
TIMEZONE = "Asia/Ho_Chi_Minh"

FACE_RECOGNITION_MODEL = "facenet"  # Options: "insightface", "facenet"
ATTENDANCE_LOG = "attendance_log.csv"
API_CALL_DELAY = 30

# FaceNet Recognition Configuration
FACENET_PRETRAINED_MODEL_PATH = "./pretrain_models/facenet"
FACENET_PRETRAINED_MODEL = "vggface2"
FACENET_RECOGNITION_THRESHOLD = 0.5
FACENET_SIMILARITY_THRESHOLD = 0.8
FACE_EMBEDDING_FACENET = "all_embeddings_facenet.pkl"
FACE_EMBEDDING_METADATA_FACENET = "metadata_facenet.json"

# InsightFace Model Configuration
INSIGHTFACE_PRETRAINED_MODEL_PATH = "./pretrain_models/insightface"
INSIGHTFACE_PRETRAINED_MODEL = "buffalo_l"
INSIGHTFACE_ALLOWED_MODULES = ["detection", "recognition"] # detection, recognition, landmark, genderage
INSIGHTFACE_DET_THRESH = 0.3
INSIGHTFACE_DET_SIZE = (640, 640)
INSIGHTFACE_RECOGNITION_THRESHOLD = 2
INSIGHTFACE_SIMILARITY_THRESHOLD = 0.75
FACE_EMBEDDING_INSIGHTFACE = "all_embeddings_insightface.pkl"
FACE_EMBEDDING_METADATA_INSIGHTFACE = "metadata_insightface.json"
