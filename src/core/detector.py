"""
Object Detection Core Module
Handles all computer vision detection operations
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import json

class ObjectDetector:
    """Core object detection engine using YOLO models"""

    def __init__(self, model_type='yolov8n', confidence=0.45, device='cpu'):
        """
        Initialize the detector

        Args:
            model_type: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l)
            confidence: Confidence threshold for detections
            device: Device to run inference on (cpu, cuda)
        """
        self.model_type = model_type
        self.confidence = confidence
        self.device = device
        self.model = None
        self.is_initialized = False
        self.detection_count = 0
        self.current_detections = []
        self.lock = threading.Lock()

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the YOLO model"""
        try:
            model_path = f"{self.model_type}.pt"
            self.model = YOLO(model_path)

            # Move to appropriate device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
                print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                if self.device == 'cuda':
                    print("CUDA requested but not available, falling back to CPU")
                self.device = 'cpu'
                print("Model loaded on CPU")

            self.is_initialized = True

        except Exception as e:
            print(f"Error initializing model: {e}")
            self.is_initialized = False

    def detect(self, frame):
        """
        Perform object detection on a frame

        Args:
            frame: Input frame (numpy array)

        Returns:
            List of detection dictionaries
        """
        if not self.is_initialized or frame is None:
            return []

        try:
            # Run inference
            results = self.model(frame, conf=self.confidence, verbose=False)

            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = self.model.names[cls]

                        detection = {
                            'class': name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)

            # Update statistics
            with self.lock:
                self.current_detections = detections
                self.detection_count += len(detections)

            return detections

        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def draw_detections(self, frame, detections):
        """
        Draw detection boxes on frame

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']

            # Color based on class
            color = self._get_class_color(cls)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{cls} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20

            cv2.rectangle(annotated, (x1, label_y - label_size[1] - 4),
                         (x1 + label_size[0], label_y), color, -1)
            cv2.putText(annotated, label, (x1, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def _get_class_color(self, class_name):
        """Get color for a specific class"""
        colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),          # Blue
            'truck': (255, 0, 0),        # Blue
            'chair': (0, 165, 255),      # Orange
            'forklift': (0, 0, 255),     # Red
            'helmet': (255, 255, 0),     # Cyan
            'vest': (0, 255, 255),       # Yellow
        }
        return colors.get(class_name, (128, 128, 128))  # Gray for unknown

    def filter_by_class(self, detections, class_names):
        """
        Filter detections by class names

        Args:
            detections: List of detections
            class_names: List of class names to keep

        Returns:
            Filtered detections
        """
        return [d for d in detections if d['class'] in class_names]

    def filter_by_confidence(self, detections, min_confidence):
        """
        Filter detections by minimum confidence

        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered detections
        """
        return [d for d in detections if d['confidence'] >= min_confidence]

    def filter_by_area(self, detections, min_area=None, max_area=None):
        """
        Filter detections by bounding box area

        Args:
            detections: List of detections
            min_area: Minimum area threshold
            max_area: Maximum area threshold

        Returns:
            Filtered detections
        """
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)

            if min_area and area < min_area:
                continue
            if max_area and area > max_area:
                continue

            filtered.append(det)

        return filtered

    def get_detection_count(self):
        """Get total detection count"""
        with self.lock:
            return self.detection_count

    def get_current_detections(self):
        """Get current frame detections"""
        with self.lock:
            return self.current_detections.copy()

    def is_ready(self):
        """Check if detector is ready"""
        return self.is_initialized

    def update_confidence(self, confidence):
        """Update confidence threshold"""
        self.confidence = confidence

    def reset_stats(self):
        """Reset detection statistics"""
        with self.lock:
            self.detection_count = 0
            self.current_detections = []


class SpecializedDetector(ObjectDetector):
    """Extended detector with specialized industrial detection capabilities"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alert_zones = []
        self.tracking_enabled = False
        self.track_history = {}

    def add_alert_zone(self, zone_id, polygon_points):
        """
        Add an alert zone for monitoring

        Args:
            zone_id: Unique zone identifier
            polygon_points: List of (x, y) points defining the zone
        """
        self.alert_zones.append({
            'id': zone_id,
            'polygon': np.array(polygon_points, np.int32)
        })

    def check_zone_intrusion(self, detections):
        """
        Check if any detection is within alert zones

        Args:
            detections: List of detections

        Returns:
            List of zone intrusion alerts
        """
        alerts = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            for zone in self.alert_zones:
                if cv2.pointPolygonTest(zone['polygon'], center, False) >= 0:
                    alerts.append({
                        'zone_id': zone['id'],
                        'detection': det,
                        'timestamp': datetime.now().isoformat()
                    })

        return alerts

    def detect_ppe(self, frame, person_bbox):
        """
        Detect Personal Protective Equipment on a person

        Args:
            frame: Input frame
            person_bbox: Person bounding box [x1, y1, x2, y2]

        Returns:
            PPE detection results
        """
        x1, y1, x2, y2 = person_bbox
        person_roi = frame[y1:y2, x1:x2]

        if person_roi.size == 0:
            return {'helmet': False, 'vest': False}

        # Simple color-based PPE detection
        hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)

        # Helmet detection (usually white or yellow)
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        helmet_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # Check top portion for helmet
        top_portion = helmet_mask[:person_roi.shape[0]//3, :]
        helmet_ratio = np.sum(top_portion > 0) / top_portion.size
        has_helmet = helmet_ratio > 0.1

        # Vest detection (usually orange or yellow)
        orange_lower = np.array([5, 100, 100])
        orange_upper = np.array([15, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        vest_mask = cv2.bitwise_or(orange_mask, yellow_mask)

        # Check middle portion for vest
        middle_start = person_roi.shape[0]//3
        middle_end = 2 * person_roi.shape[0]//3
        middle_portion = vest_mask[middle_start:middle_end, :]
        vest_ratio = np.sum(middle_portion > 0) / middle_portion.size
        has_vest = vest_ratio > 0.15

        return {'helmet': has_helmet, 'vest': has_vest}

    def count_objects_in_region(self, detections, region):
        """
        Count objects within a specific region

        Args:
            detections: List of detections
            region: Region polygon points

        Returns:
            Count dictionary by class
        """
        counts = {}
        region_poly = np.array(region, np.int32)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if cv2.pointPolygonTest(region_poly, center, False) >= 0:
                cls = det['class']
                counts[cls] = counts.get(cls, 0) + 1

        return counts