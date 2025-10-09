
"""
ASL Alphabet CNN Model Inference Script
Real-time inference for ASL alphabet recognition using webcam
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse

class ASLInference:
    def __init__(self, model_path, img_size=(200, 200), confidence_threshold=0.7):
        """
        Initialize ASL inference system

        Args:
            model_path: Path to trained model file
            img_size: Input image size for model
            confidence_threshold: Minimum confidence for prediction
        """
        self.model_path = model_path
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        # Load trained model
        self.model = self.load_model()

        # ASL alphabet classes
        self.classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

        # For smoothing predictions
        self.prediction_history = deque(maxlen=10)

        print(f"ASL Inference initialized with model: {model_path}")
        print(f"Classes: {self.classes}")

    def load_model(self):
        """
        Load the trained model
        """
        try:
            model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def preprocess_image(self, image):
        """
        Preprocess image for model prediction

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Preprocessed image ready for model input
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.img_size)

        # Normalize pixel values
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)

        return image_batch

    def predict_gesture(self, image):
        """
        Predict ASL gesture from image

        Args:
            image: Input image

        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            return "No Model", 0.0

        # Preprocess image
        processed_image = self.preprocess_image(image)

        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)

        # Get class with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]

        # Convert to class name
        predicted_class = self.classes[predicted_class_idx]

        return predicted_class, confidence

    def smooth_predictions(self, prediction, confidence):
        """
        Smooth predictions over time to reduce noise

        Args:
            prediction: Current prediction
            confidence: Current confidence

        Returns:
            Smoothed prediction
        """
        # Add current prediction to history
        self.prediction_history.append((prediction, confidence))

        # Count occurrences of each prediction
        prediction_counts = {}
        total_confidence = {}

        for pred, conf in self.prediction_history:
            if conf > self.confidence_threshold:
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                total_confidence[pred] = total_confidence.get(pred, 0) + conf

        if not prediction_counts:
            return "Unknown", 0.0

        # Get most frequent prediction
        most_frequent = max(prediction_counts.keys(), 
                          key=lambda x: prediction_counts[x])

        avg_confidence = total_confidence[most_frequent] / prediction_counts[most_frequent]

        return most_frequent, avg_confidence

    def extract_hand_region(self, frame):
        """
        Extract hand region from frame using simple background subtraction

        Args:
            frame: Input frame

        Returns:
            Processed frame with hand region highlighted
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply Gaussian blur to smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Apply mask to original frame
        result = cv2.bitwise_and(frame, frame, mask=mask)

        return result, mask

    def draw_prediction_overlay(self, frame, prediction, confidence):
        """
        Draw prediction overlay on frame

        Args:
            frame: Input frame
            prediction: Predicted class
            confidence: Prediction confidence

        Returns:
            Frame with overlay
        """
        height, width = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()

        # Draw prediction box
        box_height = 100
        cv2.rectangle(overlay, (0, 0), (width, box_height), (0, 0, 0), -1)

        # Blend overlay with original frame
        alpha = 0.7
        frame_with_overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw prediction text
        text = f"Prediction: {prediction}"
        confidence_text = f"Confidence: {confidence:.2f}"

        cv2.putText(frame_with_overlay, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_with_overlay, confidence_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw confidence bar
        bar_width = int((width - 40) * confidence)
        cv2.rectangle(frame_with_overlay, (20, 80), (20 + bar_width, 90), 
                     (0, 255, 0), -1)
        cv2.rectangle(frame_with_overlay, (20, 80), (width - 20, 90), 
                     (255, 255, 255), 2)

        return frame_with_overlay

    def run_inference(self, camera_id=0, roi_size=300):
        """
        Run real-time inference on webcam feed

        Args:
            camera_id: Camera device ID (0 for default webcam)
            roi_size: Size of region of interest for hand detection
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print("Starting real-time ASL recognition...")
        print("Press 'q' to quit, 'r' to reset prediction history, 's' to save frame")

        frame_count = 0
        fps_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # Define ROI (Region of Interest) for hand detection
            roi_x = (width - roi_size) // 2
            roi_y = (height - roi_size) // 2

            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), 
                         (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
            cv2.putText(frame, "Place hand here", (roi_x, roi_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Extract ROI
            roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]

            if roi.size > 0:
                # Predict gesture on ROI
                prediction, confidence = self.predict_gesture(roi)

                # Apply smoothing
                smooth_pred, smooth_conf = self.smooth_predictions(prediction, confidence)

                # Draw prediction overlay
                frame = self.draw_prediction_overlay(frame, smooth_pred, smooth_conf)

                # Show processed ROI in corner
                roi_display = cv2.resize(roi, (150, 150))
                frame[10:160, width-160:width-10] = roi_display
                cv2.rectangle(frame, (width-160, 10), (width-10, 160), (255, 255, 255), 2)
                cv2.putText(frame, "ROI", (width-150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time

            if frame_count > 30:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display frame
            cv2.imshow('ASL Recognition', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.prediction_history.clear()
                print("Prediction history reset")
            elif key == ord('s'):
                cv2.imwrite(f'asl_frame_{int(time.time())}.jpg', frame)
                print("Frame saved")

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Inference stopped")

    def predict_image(self, image_path):
        """
        Predict ASL gesture from image file

        Args:
            image_path: Path to image file

        Returns:
            Prediction results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None

        # Make prediction
        prediction, confidence = self.predict_gesture(image)

        # Display result
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image\nPrediction: {prediction} (Conf: {confidence:.3f})')
        plt.axis('off')

        # Preprocessed image
        plt.subplot(1, 2, 2)
        processed = self.preprocess_image(image)[0]
        plt.imshow(processed)
        plt.title('Preprocessed for Model')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'prediction_{prediction}_{int(confidence*100)}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()

        return prediction, confidence

def main():
    """
    Main inference function
    """
    parser = argparse.ArgumentParser(description='ASL Alphabet Recognition Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image file for single prediction')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold (default: 0.7)')

    args = parser.parse_args()

    # Initialize inference system
    asl_inference = ASLInference(
        model_path=args.model,
        confidence_threshold=args.confidence
    )

    if args.image:
        # Single image prediction
        result = asl_inference.predict_image(args.image)
        if result:
            prediction, confidence = result
            print(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
    else:
        # Real-time inference
        asl_inference.run_inference(camera_id=args.camera)

if __name__ == "__main__":
    # For direct execution without command line arguments
    # Uncomment and modify the following lines:

    # asl_inference = ASLInference(
    #     model_path='best_asl_model.h5',  # Path to your trained model
    #     confidence_threshold=0.7
    # )
    # asl_inference.run_inference(camera_id=0)

    main()
