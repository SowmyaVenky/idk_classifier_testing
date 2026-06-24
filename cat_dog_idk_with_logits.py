import torch
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import torch.nn as nn

if __name__ == '__main__':
    paths_to_eval = [
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\images_to_classify\train\Cat\127.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\images_to_classify\train\Dog\101.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\testing\African_Bush_Elephant.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\testing\elephant-1024x691.jpg"
    ]

    # ==========================================
    # 1. SETUP STAGE 1 MODEL (Your Cat/Dog Model)
    # ==========================================
    MODEL_PATH = 'resnet18_cat_dog.pth'
    model_stage_1 = models.resnet18() 
    num_features = model_stage_1.fc.in_features
    model_stage_1.fc = nn.Linear(num_features, 2)
    model_stage_1.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model_stage_1.eval() 
    stage_1_classes = ['Cat', 'Dog']
    
    # The Logit threshold for "I Don't Know". 
    # You will need to tune this number based on your specific model's output.
    IDK_THRESHOLD = 2.5 

    # ==========================================
    # 2. SETUP STAGE 2 MODEL (Generalist Fallback)
    # ==========================================
    # We load a pre-trained ResNet-18 that knows 1000 categories (ImageNet)
    weights_stage_2 = ResNet18_Weights.DEFAULT
    model_stage_2 = resnet18(weights=weights_stage_2)
    model_stage_2.eval()
    
    # Automatically get the 1000 category names (e.g., "African elephant")
    stage_2_classes = weights_stage_2.meta["categories"] 

    # ==========================================
    # 3. PREPROCESSING
    # ==========================================
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ==========================================
    # 4. RUN THE CASCADE
    # ==========================================
    for image_path in paths_to_eval:
        filename = image_path.split('\\')[-1]
        print(f"--- Analyzing: {filename} ---")
        
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img)
        img_batch = img_tensor.unsqueeze(0) 

        # --- RUN STAGE 1 ---
        with torch.no_grad():
            output_1 = model_stage_1(img_batch)
            logits_1 = output_1[0]
            
        probabilities_1 = torch.nn.functional.softmax(logits_1, dim=0)
        category_id_1 = probabilities_1.argmax().item()
        max_logit_1 = logits_1[category_id_1].item()
        
        # Check if Stage 1 is confident enough
        if max_logit_1 >= IDK_THRESHOLD:
            # Stage 1 is confident! Stop here.
            score = probabilities_1[category_id_1].item()
            pred_class = stage_1_classes[category_id_1]
            print(f"[Stage 1]: Confident prediction -> {pred_class} (Conf: {score:.2%}, Logit: {max_logit_1:.2f})\n")
            
        else:
            # Stage 1 says IDK. Trigger the cascade!
            print(f"[Stage 1]: IDK Triggered (Logit {max_logit_1:.2f} is below threshold {IDK_THRESHOLD}). Cascading to Stage 2...")
            
            # --- RUN STAGE 2 ---
            with torch.no_grad():
                output_2 = model_stage_2(img_batch)
                
            probabilities_2 = torch.nn.functional.softmax(output_2[0], dim=0)
            category_id_2 = probabilities_2.argmax().item()
            
            score_2 = probabilities_2[category_id_2].item()
            pred_class_2 = stage_2_classes[category_id_2]
            print(f"[Stage 2]: Generalist prediction -> {pred_class_2.title()} (Conf: {score_2:.2%})\n")