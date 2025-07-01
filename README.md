# Traffic Light Detection and Recognition System

A comprehensive traffic light detection and recognition system using YOLOv8 deep learning model. This project can detect and classify traffic lights in images and videos with high accuracy.

## 🚦 Features

- **Real-time Detection**: Detect traffic lights in images and videos
- **State Classification**: Classify traffic light states (Red, Yellow, Green)
- **Custom Training**: Train your own YOLOv8 model with custom datasets
- **Web Interface**: User-friendly web interface for testing
- **Batch Processing**: Process multiple files at once
- **High Accuracy**: Leverages YOLOv8's state-of-the-art object detection

## 🛠️ Technology Stack

### Backend
- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- PyTorch

### Frontend
- Next.js 14
- React
- TypeScript
- Tailwind CSS
- shadcn/ui

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Node.js 18+ (for web interface)
- 4GB+ RAM
- GPU recommended for training (optional)

### Python Dependencies
\`\`\`
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0
PyYAML>=6.0
requests>=2.31.0
matplotlib>=3.7.0
\`\`\`

## 🚀 Quick Start

### 1. Environment Setup
\`\`\`bash
# Clone the repository
git clone <repository-url>
cd traffic-light-detection

# Run setup script
python scripts/setup_environment.py
\`\`\`

### 2. Install Dependencies
\`\`\`bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for web interface)
npm install
\`\`\`

### 3. Basic Usage

#### Image Detection
```python
from scripts.detect_traffic_lights import TrafficLightDetector

detector = TrafficLightDetector()
detections, annotated_img = detector.detect_image("path/to/image.jpg")
