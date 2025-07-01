from ultralytics import YOLO
import os
import yaml
from pathlib import Path

class TrafficLightTrainer:
    def __init__(self):
        self.model = None
        self.data_config = None
        
    def create_dataset_config(self):
        """Create dataset configuration file"""
        config = {
            'path': './dataset',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 4,  # number of classes
            'names': ['red_light', 'yellow_light', 'green_light', 'traffic_light']
        }
        
        # Create dataset directory structure
        os.makedirs('dataset/images/train', exist_ok=True)
        os.makedirs('dataset/images/val', exist_ok=True)
        os.makedirs('dataset/images/test', exist_ok=True)
        os.makedirs('dataset/labels/train', exist_ok=True)
        os.makedirs('dataset/labels/val', exist_ok=True)
        os.makedirs('dataset/labels/test', exist_ok=True)
        
        # Save config file
        with open('dataset/data.yaml', 'w') as f:
            yaml.dump(config, f)
        
        self.data_config = 'dataset/data.yaml'
        print("Dataset configuration created!")
        return config
    
    def prepare_training_data(self):
        """Prepare training data - placeholder for data preparation"""
        print("Data preparation steps:")
        print("1. Collect traffic light images")
        print("2. Annotate images using tools like LabelImg or Roboflow")
        print("3. Convert annotations to YOLO format")
        print("4. Split data into train/val/test sets")
        print("5. Place images and labels in respective folders")
        
        # Create sample annotation format documentation
        sample_annotation = """
# YOLO Annotation Format (one line per object):
# class_id center_x center_y width height (all normalized 0-1)
# Example for red traffic light:
0 0.5 0.3 0.1 0.2
        """
        
        with open('dataset/annotation_format.txt', 'w') as f:
            f.write(sample_annotation)
    
    def train_model(self, epochs=100, img_size=640, batch_size=16):
        """Train the YOLOv8 model"""
        if not os.path.exists('dataset/data.yaml'):
            print("Creating dataset configuration...")
            self.create_dataset_config()
        
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Start with nano model for faster training
        
        # Train the model
        try:
            results = self.model.train(
                data='dataset/data.yaml',
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                name='traffic_light_detection',
                project='runs/detect',
                save=True,
                plots=True,
                device='cpu'  # Change to 'cuda' if GPU available
            )
            
            print("Training completed!")
            print(f"Best model saved at: {results.save_dir}")
            
            # Save the best model to models directory
            os.makedirs('models', exist_ok=True)
            best_model_path = f"{results.save_dir}/weights/best.pt"
            if os.path.exists(best_model_path):
                import shutil
                shutil.copy(best_model_path, 'models/traffic_light_yolov8.pt')
                print("Model copied to models/traffic_light_yolov8.pt")
            
            return results
            
        except Exception as e:
            print(f"Training error: {e}")
            print("Make sure you have training data in the dataset folder")
            return None
    
    def validate_model(self):
        """Validate the trained model"""
        if self.model is None:
            model_path = 'models/traffic_light_yolov8.pt'
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                print("No trained model found!")
                return None
        
        # Validate the model
        results = self.model.val(data='dataset/data.yaml')
        return results

def main():
    """Main training function"""
    trainer = TrafficLightTrainer()
    
    print("Traffic Light Detection Model Training")
    print("=====================================")
    
    # Create dataset structure
    trainer.create_dataset_config()
    trainer.prepare_training_data()
    
    # Check if training data exists
    train_images = Path('dataset/images/train')
    if not any(train_images.glob('*.jpg')) and not any(train_images.glob('*.png')):
        print("\nWARNING: No training images found!")
        print("Please add training images and labels before running training.")
        print("You can download traffic light datasets from:")
        print("- Roboflow Universe")
        print("- Open Images Dataset")
        print("- COCO Dataset (filter for traffic lights)")
        return
    
    # Start training
    print("\nStarting training...")
    results = trainer.train_model(epochs=50, batch_size=8)  # Reduced for demo
    
    if results:
        print("\nValidating model...")
        trainer.validate_model()

if __name__ == "__main__":
    main()
