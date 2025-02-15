import torch
import torchvision
from torchvision import transforms
from PIL import Image
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

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image like ImageNet data
    ])
    return transform(image).unsqueeze(0), image  # Add batch dimension

# Perform object detection
def detect_objects(model, image_tensor, confidence_threshold=0.3):  # Reduced threshold for debugging
    with torch.no_grad():
        predictions = model(image_tensor)  # Make predictions on the input image

    # Filter predictions by confidence threshold
    scores = predictions[0]['scores']
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']

    # Keep predictions with score higher than the threshold
    keep = scores >= confidence_threshold
    return boxes[keep], scores[keep], labels[keep]

# Visualize the detections
def visualize_detections(image, boxes, labels, scores, class_names):
    fig, ax = plt.subplots(1, figsize=(12, 8))  # Create a plot
    ax.imshow(image)  # Show the image

    # Loop through all the detected boxes and draw them on the image
    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min

        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # Map label index to the corresponding class name
        class_name = class_names[label.item()] if label.item() < len(class_names) else "Unknown"

        # Print message for detection in the console
        print(f"The Object {class_name} is Detected with a score of {score:.2f}")

        # Add label and score to the bounding box
        ax.text(
            x_min, y_min - 10, f'{class_name}, Score: {score:.2f}',
            color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5)
        )

    plt.axis('off')  # Hide axis
    plt.show()  # Show the image with detections

# In the main function, make sure to set num_classes to 30 (29 classes + 1 for background)
def main():
    # Define the class names for your FOD detection task, including "background" as the first class
    class_names = [
        "background",  # The background class should always be the first in the list
        "AdjustableWrench", "Battery", "BoltNutSet",
        "BoltWasher", "Bolt", "ClampPart",
        "Cutter", "FuelCap", "Hammer",
        "Label", "LuggagePart", "LuggageTag",
        "MetalPart", "Nail", "Nut",
        "PaintChip", "Pen", "PlasticPart",
        "Pliers", "Rock", "Screwdriver",
        "Screw", "SodaCan", "Tape",
        "Washer", "Wire", "Wood",
        "Wrench"
    ]


    # Path to the trained Faster R-CNN model
    model_path = "D:\\FOD\\FOD_Detection\\fasterrcnn_fod_detection.pth"  # Update with your model path

    # Image path for detection
    image_path = "D:\\FOD\\pythonProject\\test\\001021_jpg.rf.f2594e4224c3441889d58f1bfe200500.jpg"  # Update with your image path
    # image_path = "D:\\FOD\\pythonProject\\test\\027410_jpg.rf.bc1eaf6641cf88cadca2ba52fd3fcdb4.jpg"
    # image_path = "D:\\FOD\\pythonProject\\test\\029848_jpg.rf.2c31944c91029ef9f5c02a669e5c0ed5.jpg"
    # image_path = "D:\\FOD\\pythonProject\\Screenshot 2024-12-15 175713.png"

    # Set the correct number of classes (29 object classes + 1 background class)
    num_classes = 30  # Update this if needed (29 + 1 for background)

    # Load the model
    model = load_model(model_path, num_classes)

    # Preprocess the image
    image_tensor, original_image = preprocess_image(image_path)

    # Perform object detection
    boxes, scores, labels = detect_objects(model, image_tensor)

    # If detections are found, visualize them
    if len(boxes) > 0:
        visualize_detections(original_image, boxes, labels, scores, class_names)
    else:
        print("No objects detected in the image.")

if __name__ == "__main__":
    main()
