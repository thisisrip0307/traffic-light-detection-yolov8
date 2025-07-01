import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import json

class TrafficLightDetector:
    def __init__(self, model_path="models/traffic_light_yolov8.pt"):
        """Initialize the traffic light detector with YOLOv8 model"""
        self.model_path = model_path
        self.model = None
        self.class_names = ['red_light', 'yellow_light', 'green_light', 'traffic_light']
        self.colors = {
            'red_light': (0, 0, 255),
            'yellow_light': (0, 255, 255),
            'green_light': (0, 255, 0),
            'traffic_light': (255, 0, 0)
        }
        self.load_model()
    
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"Custom model loaded from {self.model_path}")
            else:
                # Use pre-trained YOLOv8 model and fine-tune for traffic lights
                self.model = YOLO('yolov8n.pt')
                print("Using pre-trained YOLOv8 model")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = YOLO('yolov8n.pt')
    
    def detect_image(self, image_path, output_path=None, confidence=0.5):
        """Detect traffic lights in an image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run detection
        results = self.model(image, conf=confidence)
        
        # Process results
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence_score = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[class_id] if class_id < len(self.model.names) else 'unknown'
                    
                    # Filter for traffic light related classes
                    if any(keyword in class_name.lower() for keyword in ['light', 'traffic', 'signal']):
                        # Determine traffic light state based on position and color
                        roi = image[int(y1):int(y2), int(x1):int(x2)]
                        light_state = self.classify_traffic_light_state(roi)
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence_score),
                            'class': class_name,
                            'light_state': light_state
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        color = self.colors.get(light_state, (255, 255, 255))
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"{light_state}: {confidence_score:.2f}"
                        cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        if output_path:
            cv2.imwrite(output_path, annotated_image)
        
        return detections, annotated_image
    
    def classify_traffic_light_state(self, roi):
        """Classify traffic light state based on color analysis"""
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])
        
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Determine dominant color
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        
        if max_pixels < 10:  # Threshold for minimum pixels
            return 'traffic_light'
        
        if red_pixels == max_pixels:
            return 'red_light'
        elif yellow_pixels == max_pixels:
            return 'yellow_light'
        elif green_pixels == max_pixels:
            return 'green_light'
        else:
            return 'traffic_light'
    
    def detect_video(self, video_path, output_path=None, confidence=0.5):
        """Detect traffic lights in a video"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection on frame
            results = self.model(frame, conf=confidence)
            
            # Process results
            detections = []
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence_score = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        class_name = self.model.names[class_id] if class_id < len(self.model.names) else 'unknown'
                        
                        if any(keyword in class_name.lower() for keyword in ['light', 'traffic', 'signal']):
                            roi = frame[int(y1):int(y2), int(x1):int(x2)]
                            light_state = self.classify_traffic_light_state(roi)
                            
                            detection = {
                                'frame': frame_count,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence_score),
                                'class': class_name,
                                'light_state': light_state
                            }
                            detections.append(detection)
                            
                            # Draw bounding box
                            color = self.colors.get(light_state, (255, 255, 255))
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            
                            # Draw label
                            label = f"{light_state}: {confidence_score:.2f}"
                            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            frame_detections.append(detections)
            
            # Write frame to output video
            if out:
                out.write(annotated_frame)
            
            frame_count += 1
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        return frame_detections

def main():
    """Main function for testing"""
    detector = TrafficLightDetector()
    
    # Test with sample image
    test_image = "test_images/traffic_light_sample.jpg"
    if os.path.exists(test_image):
        try:
            detections, annotated_img = detector.detect_image(test_image, "output/detected_image.jpg")
            print(f"Detected {len(detections)} traffic lights")
            for detection in detections:
                print(f"- {detection['light_state']}: {detection['confidence']:.2f}")
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print(f"Test image not found: {test_image}")

if __name__ == "__main__":
    main()
