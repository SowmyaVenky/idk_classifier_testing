'''
This script implements a cascade system using ResNet models (18, 34, 152) with Random Forest classifiers as gatekeepers.
The Random Forests are trained on a calibration dataset (ImageNetV2) to predict whether the current model's prediction 
is likely correct or not.
If the Random Forest predicts that the current model is likely incorrect, the input is passed to the next model in the cascade.
This follows the paper that Ronit and team published.
'''
import numpy as np
import torch
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torchvision.transforms as transforms
from imagenetv2_pytorch import ImageNetV2Dataset
import time

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the backbone models
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()

def train_routing_classifiers(calibration_loader):
    """
    Collects logits and trains Random Forests to predict if a model is CORRECT.
    Target 1: Model was correct (Keep prediction)
    Target 0: Model was incorrect (Skip to next stage)
    """
    r18_features, r18_targets = [], []
    r34_features, r34_targets = [], []
    
    print("Extracting features from calibration data...")
    with torch.no_grad():
        for images, labels in calibration_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Extract ResNet18 outputs
            out18 = resnet18(images)
            pred18 = out18.argmax(dim=1)
            correct18 = (pred18 == labels).cpu().numpy()
            
            r18_features.append(out18.cpu().numpy())
            r18_targets.append(correct18)
            
            # Extract ResNet34 outputs
            out34 = resnet34(images)
            pred34 = out34.argmax(dim=1)
            correct34 = (pred34 == labels).cpu().numpy()
            
            r34_features.append(out34.cpu().numpy())
            r34_targets.append(correct34)

    # Flatten collected arrays
    X_r18 = np.vstack(r18_features)
    y_r18 = np.concatenate(r18_targets)
    X_r34 = np.vstack(r34_features)
    y_r34 = np.concatenate(r34_targets)
    
    print("Training Random Forest 1 (ResNet18 regulator)...")
    rf_stage1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_stage1.fit(X_r18, y_r18)
    
    print("Training Random Forest 2 (ResNet34 regulator)...")
    rf_stage2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_stage2.fit(X_r34, y_r34)
    
    return rf_stage1, rf_stage2


class RandomForestResNetCascade(nn.Module):
    def __init__(self, rf_stage1, rf_stage2):
        super(RandomForestResNetCascade, self).__init__()
        # Trained Random Forest routing models
        self.rf_stage1 = rf_stage1
        self.rf_stage2 = rf_stage2
        
        # Deep learning backbones
        self.stage1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
        self.stage2 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()
        self.stage3 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device).eval()
        
        # Performance logging counters
        self.total_images = 0
        self.s2_count = 0
        self.s3_count = 0

    def forward(self, x):
        batch_size = x.size(0)
        self.total_images += batch_size
        
        with torch.no_grad():
            # --- STAGE 1: ResNet18 Processing ---
            out1 = self.stage1(x)
            logits1_np = out1.cpu().numpy()
            
            # Predict if ResNet18 is correct (1 = Correct, 0 = Incorrect/Skip)
            rf1_preds = self.rf_stage1.predict(logits1_np)
            to_s2_mask = torch.tensor(rf1_preds == 0, dtype=torch.bool, device=device)
            
            final_outputs = out1.clone()
            if not to_s2_mask.any():
                return final_outputs
                
            # --- STAGE 2: ResNet34 Processing ---
            s2_images = x[to_s2_mask]
            self.s2_count += s2_images.size(0)
            
            out2 = self.stage2(s2_images)
            logits2_np = out2.cpu().numpy()
            
            # Predict if ResNet34 is correct
            rf2_preds = self.rf_stage2.predict(logits2_np)
            to_s3_sub_mask = torch.tensor(rf2_preds == 0, dtype=torch.bool, device=device)
            
            final_outputs[to_s2_mask] = out2
            if not to_s3_sub_mask.any():
                return final_outputs
                
            # --- STAGE 3: ResNet152 Processing ---
            to_s3_indices = torch.where(to_s2_mask)[0][to_s3_sub_mask]
            s3_images = x[to_s3_indices]
            self.s3_count += s3_images.size(0)
            
            out3 = self.stage3(s3_images)
            final_outputs[to_s3_indices] = out3
            
            return final_outputs


if __name__ == '__main__':
    start_time = time.perf_counter()
    # --- Runtime Execution Loop Example ---

    # 1. Standard Dataset Setup
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load ImageNet-V2
    dataset = ImageNetV2Dataset(variant="matched-frequency", transform=preprocess)

    # For testing this code block, split your dataset or provide a dummy calibration set
    train_set, eval_set = torch.utils.data.random_split(dataset, [2000, 8000])
    calib_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=64, shuffle=False)

    # 2. Train Routing Layers
    rf1, rf2 = train_routing_classifiers(calib_loader)

    # 3. Build Intelligent Cascade
    cascade_system = RandomForestResNetCascade(rf1, rf2)
    end_time = time.perf_counter()
    print(f"Total Random Forest Training time: {end_time - start_time:.2f} seconds")
    # 4. Evaluate System
    top1_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cascade_system(images)
            
            _, preds = outputs.topk(1, dim=1)
            total_samples += labels.size(0)
            top1_correct += preds.eq(labels.view_as(preds)).sum().item()

    # 5. Output Analytics
    print(f"\nFinal Cascade Top-1 Accuracy: {top1_correct / total_samples:.4f}")
    print(f"Skipped to Stage 2 (ResNet34):  {cascade_system.s2_count / cascade_system.total_images * 100:.2f}%")
    print(f"Skipped to Stage 3 (ResNet152): {cascade_system.s3_count / cascade_system.total_images * 100:.2f}%")
    end_time = time.perf_counter()
    print(f"Total evaluation time: {end_time - start_time:.2f} seconds")