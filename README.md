# Foreign Object Debris (FOD) Detection

## Project Overview
This project aims to detect **Foreign Object Debris (FOD)** on surfaces like airport runways using deep learning techniques. The model leverages **Faster R-CNN with ResNet50-FPN** for object detection and is trained to classify **29 different types of FOD**, including tools, batteries, bolts, and other debris.

## Technical Details
- **Model:** Faster R-CNN (Region-Based Convolutional Neural Network)
- **Frameworks:** PyTorch, Torchvision
- **Preprocessing:**
  - Images are converted to **RGB**.
  - Normalized using ImageNet mean and standard deviation.
  - Transformed into tensors before feeding into the model.
- **Inference Pipeline:**
  1. Load the trained Faster R-CNN model.
  2. Preprocess input images.
  3. Detect FOD objects with a confidence threshold.
  4. Visualize results with bounding boxes and class labels.

## Detection Capabilities
The model can detect **common FOD items**, such as:
- Adjustable Wrench, Battery, Bolt, Nut, Clamp, Cutter
- Hammer, Nails, Plastic Parts, Pliers, Rocks, Screwdrivers
- Tape, Wires, Metal Parts, Labels, Paint Chips, and more

### Outputs:
- **Bounding Boxes** around detected objects.
- **Confidence Scores** indicating detection accuracy.
- **Class Labels** for identified debris.

## Implementation Setup
- Requires **pre-trained model weights** stored in `.pth` format.
- Runs inference on **input images** to detect objects.
- Outputs **annotated images** for verification.

## Business Impact
- **Enhances runway safety** by identifying hazardous debris.
- **Reduces maintenance costs** by early detection of dangerous objects.
- **Prevents aircraft damage**, improving operational efficiency.

## How to Run
1. Install required dependencies:
   ```bash
   pip install torch torchvision
   ```
2. Load the pre-trained model and run inference:
   ```python
   import torch
   from torchvision.models.detection import fasterrcnn_resnet50_fpn

   model = fasterrcnn_resnet50_fpn(pretrained=True)
   model.eval()
   ```
3. Process input images and visualize results.

## Contributors
- Yogeshwaran


