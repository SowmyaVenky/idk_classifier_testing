'''
This is the second iteration of the gatekeeper cascade system. 
It now includes all 5 ResNet models (18, 34, 50, 101, 152) and trains a logistic regression gate between each pair of models. 
The gates are trained on a calibration dataset (ImageNetV2) to predict whether the current model's prediction is likely correct or not. 
If the gate predicts that the current model is likely incorrect, the input is passed to the next model in the cascade.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from imagenetv2_pytorch import ImageNetV2Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model sequence
model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def get_model(name):
    if name == 'resnet18': return models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    if name == 'resnet34': return models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()
    if name == 'resnet50': return models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
    if name == 'resnet101': return models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device).eval()
    if name == 'resnet152': return models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device).eval()

def extract_gatekeeper_features(logits):
    probs = F.softmax(logits, dim=1)
    topk_probs, _ = torch.topk(probs, k=2, dim=1)
    top1 = topk_probs[:, 0]
    margin = topk_probs[:, 0] - topk_probs[:, 1]
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    return torch.stack([top1, margin, entropy], dim=1)

def train_gatekeepers(calib_loader):
    # Initialize models for training
    models_list = [get_model(n) for n in model_names]
    gates = []
    
    # We need 4 gates to manage transitions between 5 models
    for i in range(len(models_list) - 1):
        print(f"Training Gate {i+1} (deciding {model_names[i]} -> {model_names[i+1]})...")
        feats, targets = [], []
        with torch.no_grad():
            for images, labels in calib_loader:
                out = models_list[i](images.to(device))
                feats.append(extract_gatekeeper_features(out).cpu().numpy())
                targets.append((out.argmax(dim=1) == labels.to(device)).cpu().numpy())
        
        gate = LogisticRegression(class_weight='balanced', random_state=42)
        gate.fit(np.vstack(feats), np.concatenate(targets))
        gates.append(gate)
    return gates

class GatekeeperCascade(nn.Module):
    def __init__(self, gates):
        super().__init__()
        self.stages = nn.ModuleList([get_model(n) for n in model_names])
        self.weights = nn.ParameterList([nn.Parameter(torch.tensor(g.coef_[0], device=device)) for g in gates])
        self.biases = nn.ParameterList([nn.Parameter(torch.tensor(g.intercept_[0], device=device)) for g in gates])
        self.stats = [0] * len(self.stages)

    def forward(self, x):
        batch_size = x.size(0)
        final_outputs = torch.zeros((batch_size, 1000), device=device)
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for i in range(len(self.stages)):
            if not mask.any(): break
            
            # Run current stage for remaining items
            subset_x = x[mask]
            self.stats[i] += subset_x.size(0)
            out = self.stages[i](subset_x)
            final_outputs[mask] = out
            
            # If not the last stage, decide who continues
            if i < len(self.stages) - 1:
                feats = extract_gatekeeper_features(out)
                decision = torch.matmul(feats, self.weights[i]) + self.biases[i]
                # If decision < 0, it means model likely failed, pass to next
                next_mask = decision < 0.0
                
                # Update mask: only keep items that are passing to the NEXT stage
                new_mask = torch.zeros_like(mask)
                new_mask[mask] = next_mask
                mask = new_mask
        return final_outputs
    
    def print_stats(self):
        total = self.stats[0] # All images start at stage 1
        print(f"\n--- Cascade Traffic Distribution ---")
        for i, count in enumerate(self.stats):
            percentage = (count / total) * 100
            print(f"Stage {i+1} ({model_names[i]}): {count} images ({percentage:.2f}%)")

# --- Runtime Execution ---
if __name__ == '__main__':
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = ImageNetV2Dataset(variant="matched-frequency", transform=preprocess)
    calib_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(10000)), batch_size=64)
    
    gates = train_gatekeepers(calib_loader)
    system = GatekeeperCascade(gates)
    
    print("\nCascade deployed. Metrics will track flow through all 5 stages.")

    # 3. Evaluate
    top1_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in calib_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = system(images)
            
            _, preds = outputs.topk(1, dim=1)
            total_samples += labels.size(0)
            top1_correct += preds.eq(labels.view_as(preds)).sum().item()

    # Print accuracy and the new statistics
    print(f"\nGatekeeper Cascade Top-1 Accuracy: {top1_correct / total_samples:.4f}")
    system.print_stats()