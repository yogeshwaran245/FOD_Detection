import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

# Load the model with the correct number of classes including the background class
def load_model(model_path, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Set pretrained=False as we're loading custom weights

    # Modify the classifier to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load the trained model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess the input frame
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize frame like ImageNet data
    ])
    return transform(frame).unsqueeze(0)  # Add batch dimension

# Perform object detection
def detect_objects(model, image_tensor, confidence_threshold=0.3):
    with torch.no_grad():
        predictions = model(image_tensor)  # Make predictions on the input image

    # Filter predictions by confidence threshold
    scores = predictions[0]['scores']
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']

    # Keep predictions with score higher than the threshold
    keep = scores >= confidence_threshold
    return boxes[keep], scores[keep], labels[keep]

# Visualize detections on the frame
def visualize_detections(frame, boxes, labels, scores, class_names):
    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = box.int().tolist()

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Map label index to the corresponding class name
        class_name = class_names[label.item()] if label.item() < len(class_names) else "Unknown"

        # Print message for detection in the console
        print(f"The Object {class_name} is Detected with a score of {score:.2f}")

        # Add label and score to the bounding box
        cv2.putText(
            frame, f'{class_name}: {score:.2f}', (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

# Main function to capture video from camera and detect objects
def main():
    # Define the class names for your FOD detection task, including "background" as the first class
    class_names = [
        "background", "AdjustableWrenchtrack_idkeyframe", "Batterytrack_idkeyframe", "BoltNutSettrack_idkeyframe",
        "BoltWashertrack_idkeyframe", "Bolttrack_idkeyframe", "ClampParttrack_idkeyframe", "Cuttertrack_idkeyframe",
        "FuelCaptrack_idkeyframe", "Hammertrack_idkeyframe", "Labeltrack_idkeyframe", "LuggageParttrack_idkeyframe",
        "LuggageTagtrack_idkeyframe", "MetalParttrack_idkeyframe", "Nailtrack_idkeyframe", "Nuttrack_idkeyframe",
        "PaintChiptrack_idkeyframe", "Pentrack_idkeyframe", "PlasticParttrack_idkeyframe", "Plierstrack_idkeyframe",
        "Rocktrack_idkeyframe", "Screwdrivertrack_idkeyframe", "Screwtrack_idkeyframe", "SodaCantrack_idkeyframe",
        "Tapetrack_idkeyframe", "Washertrack_idkeyframe", "Wiretrack_idkeyframe", "Woodtrack_idkeyframe",
        "Wrenchtrack_idkeyframe"
    ]

    # Path to the trained Faster R-CNN model
    model_path = "D:\\FOD\\FOD_Detection\\fasterrcnn_fod_detection.pth"  # Update with your model path

    # Set the correct number of classes (29 object classes + 1 background class)
    num_classes = 30  # Update this if needed (29 + 1 for background)

    # Load the model
    model = load_model(model_path, num_classes)

    # Initialize camera capture (use 0 for default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        # Convert frame from BGR to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_frame(rgb_frame)

        # Perform object detection
        boxes, scores, labels = detect_objects(model, image_tensor)

        # Visualize detections on the frame
        visualize_detections(frame, boxes, labels, scores, class_names)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
