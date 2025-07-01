import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f" Error installing requirements: {e}")
        return False
    return True

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "models",
        "output",
        "test_images",
        "runs/detect"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def download_pretrained_model():
    """Download YOLOv8 pretrained model"""
    print("Downloading YOLOv8 pretrained model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download the model
        print("✅ YOLOv8 model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False
    return True

def check_gpu_availability():
    """Check if GPU is available for training"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  GPU not available. Training will use CPU (slower)")
    except ImportError:
        print("⚠️  PyTorch not installed. Cannot check GPU availability.")

def create_sample_config():
    """Create sample configuration files"""
    # Create dataset config
    dataset_config = """
# Traffic Light Detection Dataset Configuration
path: ./dataset
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 4

# Class names
names:
  0: red_light
  1: yellow_light
  2: green_light
  3: traffic_light
"""
    
    with open("dataset/data.yaml", "w") as f:
        f.write(dataset_config)
    
    # Create training config
    training_config = """
# Training Configuration
epochs: 100
batch_size: 16
img_size: 640
device: 'cpu'  # Change to 'cuda' if GPU available
workers: 4
patience: 50
save_period: 10
"""
    
    with open("training_config.yaml", "w") as f:
        f.write(training_config)
    
    print("✅ Configuration files created!")

def main():
    """Main setup function"""
    print("🚦 Traffic Light Detection System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return
    
    print(f"✅ Python version: {sys.version}")
    
    # Create directories
    create_directory_structure()
    
    # Install requirements
    if not install_requirements():
        return
    
    # Download pretrained model
    if not download_pretrained_model():
        return
    
    # Check GPU
    check_gpu_availability()
    
    # Create configs
    create_sample_config()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add training images to dataset/images/train/")
    print("2. Add corresponding labels to dataset/labels/train/")
    print("3. Run: python scripts/train_model.py (for custom training)")
    print("4. Run: python scripts/detect_traffic_lights.py (for detection)")
    print("5. Start web interface: npm install && npm run dev")

if __name__ == "__main__":
    main()
