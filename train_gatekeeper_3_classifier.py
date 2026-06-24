'''
This is the first iteration of a gatekeeper cascade system for ImageNet classification. 
It uses ResNet18, ResNet34, and ResNet152 as the three stages of the cascade. 
It uses top1 probability, margin, and entropy as the features for the gatekeepers.
The gatekeepers are trained using lightweight Logistic Regression models to decide whether to skip to the next stage based on the features extracted from the logits of the current stage.
'''
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load backbones
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()

def extract_gatekeeper_features(logits):
    """Computes a lightweight 3-dim feature vector on the GPU."""
    probs = F.softmax(logits, dim=1)
    topk_probs, _ = torch.topk(probs, k=2, dim=1)
    
    top1 = topk_probs[:, 0]
    margin = topk_probs[:, 0] - topk_probs[:, 1]
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    
    # Stack features into a (Batch, 3) tensor
    return torch.stack([top1, margin, entropy], dim=1)

def train_gatekeepers(calibration_loader):
    """Trains lightweight Logistic Regression models as gatekeepers."""
    g1_feats, g1_targets = [], []
    g2_feats, g2_targets = [], []
    
    print("Extracting gatekeeper features...")
    with torch.no_grad():
        for images, labels in calibration_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Stage 1 Features
            out18 = resnet18(images)
            feats18 = extract_gatekeeper_features(out18)
            correct18 = (out18.argmax(dim=1) == labels).cpu().numpy()
            g1_feats.append(feats18.cpu().numpy())
            g1_targets.append(correct18)
            
            # Stage 2 Features
            out34 = resnet34(images)
            feats34 = extract_gatekeeper_features(out34)
            correct34 = (out34.argmax(dim=1) == labels).cpu().numpy()
            g2_feats.append(feats34.cpu().numpy())
            g2_targets.append(correct34)

    # Train simple, blazing-fast linear gates
    gate1 = LogisticRegression(class_weight='balanced', random_state=42)
    gate1.fit(np.vstack(g1_feats), np.concatenate(g1_targets))
    
    gate2 = LogisticRegression(class_weight='balanced', random_state=42)
    gate2.fit(np.vstack(g2_feats), np.concatenate(g2_targets))
    
    return gate1, gate2

import torch.nn as nn
import torchvision.transforms as transforms
from imagenetv2_pytorch import ImageNetV2Dataset

class GatekeeperCascade(nn.Module):
    def __init__(self, gate1, gate2):
        super(GatekeeperCascade, self).__init__()
        # Load Deep Learning Backbones
        self.stage1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
        self.stage2 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()
        self.stage3 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device).eval()
        
        # Convert Scikit-Learn weights to native GPU tensors
        self.w1 = torch.tensor(gate1.coef_[0], dtype=torch.float32, device=device)
        self.b1 = torch.tensor(gate1.intercept_[0], dtype=torch.float32, device=device)
        
        self.w2 = torch.tensor(gate2.coef_[0], dtype=torch.float32, device=device)
        self.b2 = torch.tensor(gate2.intercept_[0], dtype=torch.float32, device=device)
        
        # Analytics metrics
        self.total_images = 0
        self.s2_count = 0
        self.s3_count = 0

    def should_skip(self, logits, weights, intercept):
        """Native PyTorch gatekeeper execution."""
        feats = extract_gatekeeper_features(logits)
        # Linear decision boundary: w * x + b < 0 means predict 'incorrect' (skip)
        decision = torch.matmul(feats, weights) + intercept
        return decision < 0.0

    def forward(self, x):
        batch_size = x.size(0)
        self.total_images += batch_size
        
        with torch.no_grad():
            # --- STAGE 1: ResNet18 ---
            out1 = self.stage1(x)
            to_s2_mask = self.should_skip(out1, self.w1, self.b1)
            
            final_outputs = out1.clone()
            if not to_s2_mask.any():
                return final_outputs
                
            # --- STAGE 2: ResNet34 ---
            s2_images = x[to_s2_mask]
            self.s2_count += s2_images.size(0)
            
            out2 = self.stage2(s2_images)
            to_s3_sub_mask = self.should_skip(out2, self.w2, self.b2)
            
            final_outputs[to_s2_mask] = out2
            if not to_s3_sub_mask.any():
                return final_outputs
                
            # --- STAGE 3: ResNet152 ---
            to_s3_indices = torch.where(to_s2_mask)[0][to_s3_sub_mask]
            s3_images = x[to_s3_indices]
            self.s3_count += s3_images.size(0)
            
            out3 = self.stage3(s3_images)
            final_outputs[to_s3_indices] = out3
            
            return final_outputs


if __name__ == '__main__':
    # --- Runtime Execution Loop ---

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageNetV2Dataset(variant="matched-frequency", transform=preprocess)
    train_set, eval_set = torch.utils.data.random_split(dataset, [2000, len(dataset)-2000])

    calib_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=64, shuffle=False)

    # 1. Train linear gates
    gate1, gate2 = train_gatekeepers(calib_loader)

    # 2. Deploy native GPU gatekeeper cascade
    gatekeeper_system = GatekeeperCascade(gate1, gate2)

    # 3. Evaluate
    top1_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = gatekeeper_system(images)
            
            _, preds = outputs.topk(1, dim=1)
            total_samples += labels.size(0)
            top1_correct += preds.eq(labels.view_as(preds)).sum().item()

    print(f"\nGatekeeper Cascade Top-1 Accuracy: {top1_correct / total_samples:.4f}")
    print(f"Skipped to ResNet34:  {gatekeeper_system.s2_count / gatekeeper_system.total_images * 100:.2f}%")
    print(f"Skipped to ResNet152: {gatekeeper_system.s3_count / gatekeeper_system.total_images * 100:.2f}%")
