#!/usr/bin/env python3
"""
Basic Object Detection Example
Demonstrates simple usage of the Edge Vision AI platform
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.detector import ObjectDetector
from src.core.camera import CameraManager
import cv2
import time

def main():
    """Run basic object detection"""

    # Initialize detector
    print("Initializing detector...")
    detector = ObjectDetector(
        model_type='yolov8n',
        confidence=0.45,
        device='cpu'  # Change to 'cuda' if GPU available
    )

    # Initialize camera
    print("Initializing camera...")
    camera = CameraManager(
        source=0,  # USB camera index or IP camera URL
        resolution=(640, 480),
        fps=30
    )

    # Start camera
    camera.start()
    time.sleep(2)  # Wait for camera to initialize

    print("Starting detection (press 'q' to quit)...")

    try:
        while True:
            # Get frame
            frame = camera.get_frame()

            if frame is not None:
                # Detect objects
                detections = detector.detect(frame)

                # Draw detections
                annotated_frame = detector.draw_detections(frame, detections)

                # Display results
                cv2.imshow("Edge Vision AI - Detection", annotated_frame)

                # Print detection info
                if detections:
                    print(f"Detected {len(detections)} objects:")
                    for det in detections:
                        print(f"  - {det['class']}: {det['confidence']:.2f}")

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame available")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()