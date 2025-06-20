import cv2
from vidgear.gears import CamGear
from config import CAMERA_INDEX, IS_NVIDIA_GPU
from logger import logging


class VidGearService:
    """Hardware-accelerated video capture using VidGear"""

    def __init__(self, source, options=None):
        """
        Initialize video capture with the given source

        Args:
            source: Video source (RTSP URL, file path, or camera index)
            options: VidGear options dictionary
            enable_streaming: Whether to enable output streaming 
            output_params: Parameters for StreamGear
        """
        if options is None:
            # Default options for optimal performance with RTSP
            options = {
                "THREADED_QUEUE_MODE": True,  # VidGear's internal threading
                "CAP_PROP_BUFFERSIZE": 2,     # Small buffer size for low latency
            }
        if IS_NVIDIA_GPU:
            gstreamer_source = (
                'rtspsrc location={} latency=10 protocols=udp drop-on-latency=false !'.format(source) +
                'rtph264depay !'
                'h264parse !'
                'nvv4l2decoder !'
                'nvvideoconvert ! video/x-raw, format=BGR ! '
                'appsink max-buffers=1 drop=true sync=false')
            # Initialize CamGear with source
            self.stream = CamGear(source=gstreamer_source, logging=True, backend=cv2.CAP_GSTREAMER)
        else:
            # Initialize CamGear with 
            self.stream = CamGear(source=CAMERA_INDEX, logging=True, **options)
        self.stream.start()

    def read(self):
        """
        Read a frame (compatible with OpenCV's VideoCapture.read)

        Returns:
            tuple: (success, frame)
        """
        # Get frame directly from CamGear
        frame = self.stream.read()

        # Return in OpenCV-compatible format
        return (frame is not None), frame

    def release(self):
        """Release resources"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            logging.info("CamGear stream released")
