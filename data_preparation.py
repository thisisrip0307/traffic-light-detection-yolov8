import os
import requests
import cv2
import numpy as np
from pathlib import Path
import json
import random
from urllib.parse import urlparse

class DatasetPreparer:
    def __init__(self):
        self.dataset_dir = Path('dataset')
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    def download_sample_images(self):
        """Download sample traffic light images for demonstration"""
        sample_urls = [
            "https://images.unsplash.com/photo-1544197150-b99a580bb7a8?w=640",  # Traffic light
            "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640",  # City traffic
            "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=640",  # Street view
        ]
        
        print("Downloading sample images...")
        for i, url in enumerate(sample_urls):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    filename = f"sample_{i+1}.jpg"
                    filepath = self.images_dir / 'train' / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"Downloaded: {filename}")
                    
                    # Create dummy annotation
                    self.create_dummy_annotation(filename)
                    
            except Exception as e:
                print(f"Failed to download image {i+1}: {e}")
    
    def create_dummy_annotation(self, image_filename):
        """Create dummy YOLO format annotation"""
        # Create a simple annotation file
        label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = self.labels_dir / 'train' / label_filename
        
        # Dummy annotation (you would replace this with real annotations)
        # Format: class_id center_x center_y width height (normalized)
        dummy_annotation = "3 0.5 0.3 0.1 0.2\n"  # traffic_light class
        
        with open(label_path, 'w') as f:
            f.write(dummy_annotation)
    
    def augment_images(self, input_dir, output_dir, num_augmentations=3):
        """Apply data augmentation to increase dataset size"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for img_file in input_path.glob('*.jpg'):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Original image
            base_name = img_file.stem
            cv2.imwrite(str(output_path / f"{base_name}_orig.jpg"), img)
            
            # Apply augmentations
            for i in range(num_augmentations):
                augmented = self.apply_augmentation(img)
                cv2.imwrite(str(output_path / f"{base_name}_aug_{i}.jpg"), augmented)
    
    def apply_augmentation(self, image):
        """Apply random augmentation to image"""
        h, w = image.shape[:2]
        
        # Random brightness adjustment
        brightness = random.uniform(0.7, 1.3)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Random rotation
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random noise
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train/val/test sets"""
        # This is a placeholder - in real scenario, you'd split your actual data
        print(f"Dataset split ratios:")
        print(f"Train: {train_ratio*100}%")
        print(f"Validation: {val_ratio*100}%")
        print(f"Test: {test_ratio*100}%")
    
    def validate_annotations(self):
        """Validate YOLO format annotations"""
        issues = []
        
        for split in ['train', 'val', 'test']:
            label_dir = self.labels_dir / split
            image_dir = self.images_dir / split
            
            for label_file in label_dir.glob('*.txt'):
                # Check if corresponding image exists
                img_name = label_file.stem + '.jpg'
                if not (image_dir / img_name).exists():
                    img_name = label_file.stem + '.png'
                    if not (image_dir / img_name).exists():
                        issues.append(f"Missing image for {label_file}")
                        continue
                
                # Validate annotation format
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            issues.append(f"{label_file}:{line_num} - Invalid format")
                            continue
                        
                        # Check if values are in valid range
                        class_id, cx, cy, w, h = map(float, parts)
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            issues.append(f"{label_file}:{line_num} - Values out of range")
                
                except Exception as e:
                    issues.append(f"{label_file} - Error reading file: {e}")
        
        if issues:
            print("Annotation issues found:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"- {issue}")
            if len(issues) > 10:
                print(f"... and {len(issues) - 10} more issues")
        else:
            print("All annotations are valid!")
        
        return issues

def main():
    """Main data preparation function"""
    preparer = DatasetPreparer()
    
    print("Traffic Light Dataset Preparation")
    print("=================================")
    
    # Download sample images for demonstration
    preparer.download_sample_images()
    
    # Validate annotations
    print("\nValidating annotations...")
    preparer.validate_annotations()
    
    # Split dataset info
    preparer.split_dataset()
    
    print("\nDataset preparation completed!")
    print("\nNext steps:")
    print("1. Replace sample images with real traffic light images")
    print("2. Create proper annotations using tools like LabelImg")
    print("3. Run the training script: python scripts/train_model.py")

if __name__ == "__main__":
    main()
