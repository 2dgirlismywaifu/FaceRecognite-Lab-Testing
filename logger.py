import codecs
import logging
import os
from pathlib import Path
import sys

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "timekeeper-attendance.log")
# Create the log file if it doesn't exist
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Touch the log file to create it if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'a'):
        pass

# Configure stdout to use UTF-8 (critical on Windows)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)