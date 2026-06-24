import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

CONFIDENCE_THRESHOLD = 0.75  # Tune this: raise it to be stricter

if __name__ == '__main__':
    paths_to_eval = [
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\images_to_classify\train\Cat\127.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\images_to_classify\train\Dog\101.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\testing\African_Bush_Elephant.jpg",
        r"C:\Venky\UHSummerProgram\resnet_idk_cascade\testing\elephant-1024x691.jpg"
    ]

    MODEL_PATH = 'resnet18_cat_dog.pth'
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    classes = ['Cat', 'Dog']

  # 3. Load and preprocess your image
    for image_path in paths_to_eval:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img)
        img_batch = img_tensor.unsqueeze(0) 

        # 4. Run inference
        with torch.no_grad():
            output = model(img_batch)
            logits = output[0] # Grab the raw scores BEFORE softmax

        # 5. Extract probabilities and top class
        probabilities = torch.nn.functional.softmax(logits, dim=0)
        category_id = probabilities.argmax().item()
        
        # Get the raw logit score for the winning class
        max_logit = logits[category_id].item()

        classes = ['Cat', 'Dog'] 

        # 6. Apply an "Unknown" threshold
        # You will need to test your model to find the right threshold number.
        # Try testing real cats vs elephants to find the perfect cutoff.
        LOGIT_THRESHOLD = 2.5 

        score = probabilities[category_id].item()
        category_name = classes[category_id]

        if max_logit < LOGIT_THRESHOLD:
            print(f"Image: {image_path.split('\\')[-1]}")
            print(f"Prediction: UNKNOWN (Likely OOD) - The model guessed {category_name} ({score:.2%}), but raw score was too low ({max_logit:.2f})\n")
        else:
            print(f"Image: {image_path.split('\\')[-1]}")
            print(f"Prediction: {category_name} (Confidence: {score:.2%}, Raw Score: {max_logit:.2f})\n")
