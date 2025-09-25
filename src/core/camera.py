"""
Camera Management Module
Handles multiple camera sources and frame capture
"""

import cv2
import threading
import time
import numpy as np
from datetime import datetime
import queue

class CameraManager:
    """Manages camera connections and frame capture"""

    def __init__(self, source=0, resolution=(640, 480), fps=30, buffer_size=10):
        """
        Initialize camera manager

        Args:
            source: Camera source (int for USB, string for IP camera)
            resolution: Target resolution (width, height)
            fps: Target frames per second
            buffer_size: Frame buffer size
        """
        self.source = source
        self.resolution = resolution
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap = None
        self.is_running = False
        self.is_connected = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.current_frame = None
        self.lock = threading.Lock()
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0
        
        # Thread for capture
        self.capture_thread = None

    def start(self):
        """Start camera capture"""
        if self.is_running:
            return
        
        # Initialize camera
        self._initialize_camera()
        
        if self.is_connected:
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            print(f"Camera started: {self.source}")
        else:
            print(f"Failed to start camera: {self.source}")

    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False
        print("Camera stopped")

    def _initialize_camera(self):
        """Initialize camera connection"""
        try:
            # Determine source type
            if isinstance(self.source, str):
                # IP camera
                self.cap = cv2.VideoCapture(self.source)
            else:
                # USB camera
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.is_connected = True
                
                # Read actual settings
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            else:
                print(f"Failed to open camera: {self.source}")
                self.is_connected = False
                
        except Exception as e:
            print(f"Camera initialization error: {e}")
            self.is_connected = False

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        frame_time = 1.0 / self.fps
        last_time = time.time()
        
        while self.is_running and self.cap:
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    # Update current frame
                    with self.lock:
                        self.current_frame = frame
                    
                    # Add to buffer (non-blocking)
                    try:
                        if self.frame_buffer.full():
                            self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait(frame)
                    except:
                        pass
                    
                    # Update statistics
                    self.frame_count += 1
                    
                    # Calculate actual FPS
                    current_time = time.time()
                    if current_time - last_time > 1.0:
                        self.actual_fps = self.frame_count / (current_time - last_time)
                        self.frame_count = 0
                        last_time = current_time
                else:
                    # Reconnect if disconnected
                    print("Camera disconnected, attempting to reconnect...")
                    self._reconnect()
                    
                # Frame rate limiting
                time.sleep(max(0, frame_time - (time.time() - last_time)))
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)

    def _reconnect(self):
        """Attempt to reconnect to camera"""
        if self.cap:
            self.cap.release()
        
        time.sleep(1)
        self._initialize_camera()

    def get_frame(self):
        """Get the latest frame"""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_buffered_frame(self):
        """Get frame from buffer (may have slight delay)"""
        try:
            return self.frame_buffer.get_nowait()
        except queue.Empty:
            return self.get_frame()

    def is_connected(self):
        """Check if camera is connected"""
        return self.is_connected and self.cap is not None and self.cap.isOpened()

    def get_fps(self):
        """Get actual FPS"""
        return self.actual_fps

    def get_uptime(self):
        """Get camera uptime in seconds"""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0

    def capture_snapshot(self, filename=None):
        """Capture a snapshot"""
        frame = self.get_frame()
        if frame is not None:
            if filename is None:
                filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            return filename
        return None


class MultiCameraManager:
    """Manages multiple camera sources"""

    def __init__(self, cameras):
        """
        Initialize multi-camera manager

        Args:
            cameras: List of camera configurations
                    [{'name': 'cam1', 'source': 0, 'resolution': (640, 480)}, ...]
        """
        self.cameras = {}
        
        for cam_config in cameras:
            name = cam_config.get('name', f"camera_{len(self.cameras)}")
            source = cam_config.get('source', 0)
            resolution = cam_config.get('resolution', (640, 480))
            fps = cam_config.get('fps', 30)
            
            self.cameras[name] = CameraManager(source, resolution, fps)

    def start_all(self):
        """Start all cameras"""
        for name, camera in self.cameras.items():
            camera.start()
            print(f"Started camera: {name}")

    def stop_all(self):
        """Stop all cameras"""
        for name, camera in self.cameras.items():
            camera.stop()
            print(f"Stopped camera: {name}")

    def get_frame(self, camera_name):
        """Get frame from specific camera"""
        if camera_name in self.cameras:
            return self.cameras[camera_name].get_frame()
        return None

    def get_all_frames(self):
        """Get frames from all cameras"""
        frames = {}
        for name, camera in self.cameras.items():
            frames[name] = camera.get_frame()
        return frames

    def get_status(self):
        """Get status of all cameras"""
        status = {}
        for name, camera in self.cameras.items():
            status[name] = {
                'connected': camera.is_connected(),
                'fps': camera.get_fps(),
                'uptime': camera.get_uptime()
            }
        return status