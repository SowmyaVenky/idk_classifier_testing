import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Define device and thresholds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold_resnet18 = 0.85  # Confidence required to stop at ResNet18

# 2. Load pre-trained models
model_fast = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
model_heavy = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)

model_fast_classes = models.ResNet18_Weights.DEFAULT.meta['categories']
model_heavy_classes = models.ResNet50_Weights.DEFAULT.meta['categories']

model_fast.eval()
model_heavy.eval()

# 3. Image Preprocessing Pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_with_cascade(image_path):
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # --- STAGE 1: ResNet-18 (Fast) ---
    with torch.no_grad():
        output_fast = model_fast(img_tensor)
        probabilities_fast = torch.nn.functional.softmax(output_fast, dim=1)
        max_prob_fast, predicted_class_fast = torch.max(probabilities_fast, dim=1)
    
    print(f"Stage 1 (ResNet18) finished. Predicted Class: {model_fast_classes[predicted_class_fast.item()]}, Confidence: {max_prob_fast.item():.4f}")

    # Check if ResNet18 is confident enough (not IDK)
    if max_prob_fast.item() >= threshold_resnet18:
        print(f"Stage 1 (ResNet18) confident. Predicted Class: {predicted_class_fast.item()}")
        return predicted_class_fast.item()
    
    # --- STAGE 2: ResNet-50 (Heavy/Accurate) ---
    print("Stage 1 unsure (IDK). Escalating to Stage 2 (ResNet50)...")
    with torch.no_grad():
        output_heavy = model_heavy(img_tensor)
        probabilities_heavy = torch.nn.functional.softmax(output_heavy, dim=1)
        max_prob_heavy, predicted_class_heavy = torch.max(probabilities_heavy, dim=1)
          
    print(f"Stage 2 (ResNet50) finished. Predicted Class: {model_heavy_classes[predicted_class_heavy.item()]}, Confidence: {max_prob_heavy.item():.4f}")
    return predicted_class_heavy.item()


paths_to_eval = [
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\images_to_classify\train\Cat\127.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\images_to_classify\train\Dog\101.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\testing\African_Bush_Elephant.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\testing\elephant-1024x691.jpg"
    ]

for path in paths_to_eval:
    classify_with_cascade(path)
    