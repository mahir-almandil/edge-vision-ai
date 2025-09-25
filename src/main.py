#!/usr/bin/env python3
"""
Edge Vision AI Platform - Main Application
Industrial-grade computer vision for edge devices
"""

import sys
import os
import logging
import signal
import yaml
import argparse
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import threading
import time
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import ObjectDetector
from core.camera import CameraManager
from core.alert_manager import AlertManager
from core.cloud_connector import CloudConnector
from api.routes import api_bp
from utils.logger import setup_logger
from utils.config import Config

# Global variables
app = Flask(__name__,
           template_folder='../web/templates',
           static_folder='../web/static')
config = None
detector = None
camera_manager = None
alert_manager = None
cloud_connector = None
logger = None
running = True

def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    global running
    logger.info("Shutting down Edge Vision AI...")
    running = False
    if camera_manager:
        camera_manager.stop()
    if cloud_connector:
        cloud_connector.disconnect()
    sys.exit(0)

def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        # Create default config if not exists
        default_config = {
            'camera': {
                'source': 0,
                'resolution': [640, 480],
                'fps': 30
            },
            'model': {
                'type': 'yolov8n',
                'confidence': 0.45,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'app': {
                'port': 8080,
                'debug': False,
                'enable_recording': False
            },
            'alerts': {
                'enable': False,
                'email': {
                    'enabled': False,
                    'smtp_server': '',
                    'recipients': []
                }
            },
            'cloud': {
                'provider': 'none',
                'endpoint': '',
                'topic': ''
            }
        }

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        return default_config

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_components():
    """Initialize all system components"""
    global detector, camera_manager, alert_manager, cloud_connector

    # Initialize object detector
    logger.info(f"Initializing detector with model: {config['model']['type']}")
    detector = ObjectDetector(
        model_type=config['model']['type'],
        confidence=config['model']['confidence'],
        device=config['model']['device']
    )

    # Initialize camera manager
    logger.info(f"Initializing camera: {config['camera']['source']}")
    camera_manager = CameraManager(
        source=config['camera']['source'],
        resolution=tuple(config['camera']['resolution']),
        fps=config['camera']['fps']
    )

    # Initialize alert manager if enabled
    if config['alerts']['enable']:
        logger.info("Initializing alert system")
        alert_manager = AlertManager(config['alerts'])

    # Initialize cloud connector if configured
    if config['cloud']['provider'] != 'none':
        logger.info(f"Connecting to cloud: {config['cloud']['provider']}")
        cloud_connector = CloudConnector(config['cloud'])
        cloud_connector.connect()

# Flask Routes
@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html', config=config)

@app.route('/feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while running:
            frame = camera_manager.get_frame()
            if frame is not None:
                # Run detection
                detections = detector.detect(frame)

                # Draw detections
                annotated_frame = detector.draw_detections(frame, detections)

                # Check alerts
                if alert_manager:
                    alert_manager.check_alerts(detections)

                # Send to cloud
                if cloud_connector:
                    cloud_connector.send_detections(detections)

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_data = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                time.sleep(0.1)

    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    """System status endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"

    return jsonify({
        'status': 'running',
        'camera': camera_manager.is_connected(),
        'detector': detector.is_ready(),
        'gpu': {
            'available': gpu_available,
            'device': gpu_name,
            'memory_used': torch.cuda.memory_allocated() if gpu_available else 0
        },
        'fps': camera_manager.get_fps(),
        'detections_count': detector.get_detection_count(),
        'uptime': camera_manager.get_uptime()
    })

@app.route('/api/detections')
def get_detections():
    """Get current detections"""
    return jsonify(detector.get_current_detections())

@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Configuration management"""
    if request.method == 'POST':
        new_config = request.json
        # Update configuration
        global config
        config.update(new_config)
        # Save to file
        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f)
        # Restart components with new config
        initialize_components()
        return jsonify({'status': 'updated'})
    else:
        return jsonify(config)

def main():
    """Main application entry point"""
    global config, logger

    # Parse arguments
    parser = argparse.ArgumentParser(description='Edge Vision AI Platform')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--port', type=int, default=None,
                       help='Override port from config')
    parser.add_argument('--camera', type=str, default=None,
                       help='Override camera source')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('edge-vision-ai', level=logging.DEBUG if args.debug else logging.INFO)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Override with command line arguments
    if args.port:
        config['app']['port'] = args.port
    if args.camera:
        config['camera']['source'] = args.camera if not args.camera.isdigit() else int(args.camera)
    if args.debug:
        config['app']['debug'] = True

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize components
    logger.info("Initializing Edge Vision AI Platform...")
    initialize_components()

    # Register API blueprint
    app.register_blueprint(api_bp, url_prefix='/api')

    # Start camera
    camera_manager.start()

    # Print startup info
    logger.info("="*50)
    logger.info("Edge Vision AI Platform Started")
    logger.info(f"Dashboard: http://0.0.0.0:{config['app']['port']}")
    logger.info(f"API: http://0.0.0.0:{config['app']['port']}/api")
    logger.info(f"Video Feed: http://0.0.0.0:{config['app']['port']}/feed")
    logger.info(f"Device: {config['model']['device']}")
    logger.info(f"Model: {config['model']['type']}")
    logger.info("="*50)

    # Run Flask app
    app.run(host='0.0.0.0',
           port=config['app']['port'],
           debug=config['app']['debug'],
           threaded=True)

if __name__ == '__main__':
    main()