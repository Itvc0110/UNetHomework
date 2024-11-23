import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
from segmentation_models_pytorch import UnetPlusPlus
import os

# Define color mapping for mask visualization
COLOR_DICT = {
    0: (0, 0, 0),       # Background
    1: (255, 0, 0),     # Class 1 (Red)
    2: (0, 255, 0)      # Class 2 (Green)
}

def mask_to_rgb(mask, color_dict):

    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, color in color_dict.items():
        output[mask == k] = color
    return output

def load_model(checkpoint_path, device):

    model = UnetPlusPlus(
        encoder_name="efficientnet-b6",
        encoder_weights=None,
        in_channels=3,
        classes=3  
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

def infer(model, image_path, device, output_path="segmented_image.png"):

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    tensor_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(tensor_image)
        mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()

    segmented_image = mask_to_rgb(mask, COLOR_DICT)

    # Save segmented image
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved to {output_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Infer a segmentation model on a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output_path", type=str, default="segmented_image.png", help="Path to save the output image.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint_path, device)

    infer(model, args.image_path, device, args.output_path)

if __name__ == "__main__":
    main()
