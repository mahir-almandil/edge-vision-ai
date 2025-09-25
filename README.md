# Edge Vision AI Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA Jetson](https://img.shields.io/badge/Platform-NVIDIA%20Jetson-green.svg)](https://developer.nvidia.com/embedded-computing)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A production-ready, open-source edge AI platform for real-time computer vision on industrial IoT devices. Deploy intelligent vision systems on NVIDIA Jetson, Raspberry Pi, or any edge device with live video analytics, natural language interface, and cloud connectivity.

## Features

- **Real-time Object Detection** - YOLOv8 powered detection with 80+ object classes
- **Natural Language Interface** - Chat with your camera using simple commands
- **Edge Optimized** - Runs efficiently on resource-constrained devices
- **Multi-Camera Support** - USB, IP, and CSI camera compatibility
- **Industrial Ready** - PPE detection, safety monitoring, quality control
- **Cloud Integration** - AWS IoT, Azure IoT Hub, MQTT support
- **Extensible Architecture** - Plugin system for custom detections
- **Web Dashboard** - Real-time monitoring from any browser


## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.x (for GPU acceleration, optional)
- USB/IP camera
- 4GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/edge-vision-ai.git
cd edge-vision-ai

# Install dependencies
pip install -r requirements.txt

# Download AI model
python scripts/download_models.py

# Run the application
python src/main.py
```

Access the dashboard at `http://localhost:8080`

## Docker Deployment

```bash
# Build the Docker image
docker build -t edge-vision-ai .

# Run with GPU support
docker run --gpus all -p 8080:8080 --device=/dev/video0 edge-vision-ai

# Run CPU-only
docker run -p 8080:8080 --device=/dev/video0 edge-vision-ai
```

## Industrial Use Cases

### Manufacturing
- Quality control inspection
- Assembly line monitoring
- Defect detection
- Production counting

### Safety & Security
- PPE compliance monitoring
- Restricted area surveillance
- Incident detection
- Access control

### Logistics
- Package tracking
- Inventory management
- Loading dock monitoring
- Vehicle detection

### Retail
- Customer counting
- Queue management
- Shelf monitoring
- Heat mapping

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Configuration](docs/CONFIGURATION.md)
- [API Reference](docs/API.md)
- [Plugin Development](docs/PLUGINS.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Configuration

Create a `config.yaml` file:

```yaml
# Camera Configuration
camera:
  source: 0  # USB camera index or IP camera URL
  resolution: [640, 480]
  fps: 30

# AI Model Configuration
model:
  type: "yolov8n"  # Options: yolov8n, yolov8s, yolov8m, yolov8l
  confidence: 0.45
  device: "cuda"  # Options: cuda, cpu

# Application Settings
app:
  port: 8080
  debug: false
  enable_recording: false

# Alert Configuration
alerts:
  enable: true
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    recipients: ["admin@example.com"]

# Cloud Integration (Optional)
cloud:
  provider: "none"  # Options: aws, azure, mqtt, none
  endpoint: ""
  topic: ""
```

## Architecture

```
edge-vision-ai/
├── src/
│   ├── core/           # Core detection engine
│   ├── api/            # REST API endpoints
│   ├── web/            # Web dashboard
│   ├── plugins/        # Plugin system
│   └── utils/          # Utilities
├── config/             # Configuration files
├── docker/             # Docker configurations
├── examples/           # Example implementations
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Plugin System

Create custom detection plugins:

```python
# plugins/custom_detector.py
from src.core.base_plugin import BasePlugin

class CustomDetector(BasePlugin):
    def __init__(self, config):
        super().__init__(config)

    def process_frame(self, frame):
        # Your detection logic here
        detections = self.detect_custom_objects(frame)
        return detections

    def detect_custom_objects(self, frame):
        # Implement your detection
        pass
```

## Performance

| Platform | Model | FPS | RAM Usage | Power |
|----------|-------|-----|-----------|-------|
| Jetson Nano | YOLOv8n | 15 | 2.1GB | 5W |
| Jetson Xavier NX | YOLOv8n | 45 | 3.2GB | 15W |
| Jetson AGX Orin | YOLOv8n | 100+ | 4.5GB | 30W |
| RPi 4 (8GB) | YOLOv8n | 3-5 | 1.8GB | 7W |
| Desktop GPU | YOLOv8n | 120+ | 2.5GB | 75W |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/edge-vision-ai.git
cd edge-vision-ai
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

## Examples

### Basic Object Detection

```python
from edge_vision_ai import VisionSystem

# Initialize system
vision = VisionSystem(camera_index=0)

# Start detection
vision.start(callback=lambda detections: print(f"Detected: {detections}"))
```

### Custom Alert System

```python
from edge_vision_ai import VisionSystem, AlertManager

# Configure alerts
alerts = AlertManager()
alerts.add_rule("person", threshold=5, action="email")

# Start with alerts
vision = VisionSystem(camera_index=0, alert_manager=alerts)
vision.start()
```

### Multi-Camera Setup

```python
from edge_vision_ai import MultiCameraSystem

# Configure multiple cameras
cameras = [
    {"name": "entrance", "source": 0},
    {"name": "warehouse", "source": "rtsp://192.168.1.100/stream"},
    {"name": "loading", "source": 1}
]

# Start multi-camera system
system = MultiCameraSystem(cameras)
system.start()
```

## Showcase

Organizations using Edge Vision AI:

- Manufacturing facilities for quality control
- Warehouses for inventory management
- Retail stores for customer analytics
- Construction sites for safety monitoring

*Add your implementation to our [showcase](docs/SHOWCASE.md)!*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) for edge AI platform
- [OpenCV](https://opencv.org/) for computer vision
- [Flask](https://flask.palletsprojects.com/) for web framework

## Support

- [Documentation](docs/)
- [Discussions](https://github.com/yourusername/edge-vision-ai/discussions)
- [Issue Tracker](https://github.com/yourusername/edge-vision-ai/issues)
- Email: support@edgevisionai.com

## Roadmap

- [ ] TensorRT optimization
- [ ] DeepStream integration
- [ ] Custom model training UI
- [ ] Mobile app
- [ ] Kubernetes deployment
- [ ] Edge-to-cloud pipeline
- [ ] Video analytics dashboard
- [ ] Multi-language support

---

<div align="center">
  <b>Built with for the Edge AI Community</b>
  <br>
  <a href="https://github.com/yourusername/edge-vision-ai">Star us on GitHub</a>
</div>
