'''
This script implements a gatekeeper cascade for ImageNet classification using ResNet models. 
It trains logistic regression gatekeepers to decide whether to pass an image to the next model in the cascade based on the confidence
of the current model's predictions. The cascade consists of ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152 models. 
The script evaluates the overall accuracy and prints the traffic distribution across the stages of the cascade.

This script is a clever implementation of **Cascaded Inference** (also known as "early exiting").

In simple terms, it creates a smart, cost-saving pipeline for image classification. Instead of running every single image through a massive, slow model like ResNet152, it tries to classify images using a small, fast model first (ResNet18). If the small model is confident, it accepts the answer and stops. If the small model is uncertain, a "gatekeeper" passes the image to the next, slightly larger model, and so on.

Here is a breakdown of how the code accomplishes this:

### **1. The Core Concept: Feature Extraction & Gatekeepers**

To decide if a model's prediction is trustworthy, the script looks at two things:

* **Top Probabilities:** How confident the model is between its #1 and #2 choices (extracted in `extract_gatekeeper_features`).
* **Global Average Pooling (GAP) Features:** The raw, internal representations the model built right before making its final classification.

The `train_gatekeepers` function uses a subset of data (the calibration set) to train simple **Logistic Regression** models from scikit-learn. These serve as the "gatekeepers." They learn to look at the features and probabilities and predict a simple binary outcome: *Did this model get the answer right or wrong?*

### **2. Building the PyTorch Cascade**

The `GatekeeperCascade` class is where the deployment magic happens.

Instead of keeping the gatekeepers in scikit-learn, the `__init__` function cleverly extracts the weights and biases (`g.coef_` and `g.intercept_`) from the trained Logistic Regression models and converts them into native PyTorch parameters. This allows the gatekeeper decisions to run entirely on the GPU, vastly speeding up inference.

### **3. The Forward Pass & Batch Masking**

The `forward` method is the most complex part of the script because it has to handle batches of images efficiently. You can't easily run different models on different parts of a batch simultaneously, so it uses **Boolean Masking**:

* It starts with a `mask` of all `True` values, meaning all images in the batch are passed to Stage 1 (ResNet18).
* It runs the model on the active images (`subset_x = x[mask]`).
* It computes the gatekeeper's decision using a dot product: `torch.matmul(feats, self.weights[i]) + self.biases[i]`.
* If the decision score is `< 0.0`, the gatekeeper believes the current model is wrong, and sets `next_mask` to `True` for those specific images.
* The overall `mask` is updated. In the next loop iteration (ResNet34), *only* the images that failed the previous gatekeeper are processed.

### **4. Evaluation**

Finally, the `__main__` block runs 10,000 images from the ImageNetV2 dataset through this cascade. It calculates the overall accuracy and uses `print_stats` to show the "Traffic Distribution."

If successful, you will see a distribution where the vast majority of easy images are solved by ResNet18 or ResNet34, while only a tiny percentage of the hardest images ever reach the computationally expensive ResNet152, saving massive amounts of compute time without sacrificing much accuracy.

---
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

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.fc = model.fc

    def forward(self, x):
        feats = self.features(x).flatten(1)
        logits = self.fc(feats)
        return logits, feats

def get_model(name):
    if name == 'resnet18': return FeatureExtractor(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval())        
    if name == 'resnet34': return FeatureExtractor(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval())
    if name == 'resnet50': return FeatureExtractor(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval())
    if name == 'resnet101': return FeatureExtractor(models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device).eval())
    if name == 'resnet152': return FeatureExtractor(models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device).eval())

def extract_gatekeeper_features(logits, gap_feats):
    probs = F.softmax(logits, dim=1)
    topk_probs, _ = torch.topk(probs, k=2, dim=1)
    return torch.cat([topk_probs, gap_feats], dim=1)

def train_gatekeepers(calib_loader):
    models_list = [get_model(n) for n in model_names]
    gates = []
    
    for i in range(len(models_list) - 1):
        print(f"Training Gate {i+1} (deciding {model_names[i]} -> {model_names[i+1]})...")
        feats, targets = [], []
        with torch.no_grad():
            for images, labels in calib_loader:
                out, gap = models_list[i](images.to(device))
                gate_feats = extract_gatekeeper_features(out, gap)
                feats.append(gate_feats.cpu().numpy())
                targets.append((out.argmax(dim=1) == labels.to(device)).cpu().numpy())
        
        gate = LogisticRegression(class_weight='balanced', max_iter=1000)
        gate.fit(np.vstack(feats), np.concatenate(targets))
        gates.append(gate)
    return gates

class GatekeeperCascade(nn.Module):
    def __init__(self, gates):
        super().__init__()
        self.stages = nn.ModuleList([get_model(n) for n in model_names])
        self.weights = nn.ParameterList([nn.Parameter(torch.tensor(g.coef_[0], device=device).float()) for g in gates])
        self.biases = nn.ParameterList([nn.Parameter(torch.tensor(g.intercept_[0], device=device).float()) for g in gates])
        self.stats = [0] * len(self.stages)

    def forward(self, x):
        batch_size = x.size(0)
        final_outputs = torch.zeros((batch_size, 1000), device=device)
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for i in range(len(self.stages)):
            if not mask.any(): break
            
            subset_x = x[mask]
            self.stats[i] += subset_x.size(0)
            out, gap = self.stages[i](subset_x)
            final_outputs[mask] = out
            
            if i < len(self.stages) - 1:
                feats = extract_gatekeeper_features(out, gap)
                decision = torch.matmul(feats, self.weights[i]) + self.biases[i]
                next_mask = decision < 0.0
                
                new_mask = torch.zeros_like(mask)
                new_mask[mask] = next_mask
                mask = new_mask
        return final_outputs
    
    def print_stats(self, total_samples):
        print(f"\n--- Cascade Traffic Distribution ---")
        for i, count in enumerate(self.stats):
            percentage = (count / total_samples) * 100
            print(f"Stage {i+1} ({model_names[i]}): {count} images ({percentage:.2f}%)")

if __name__ == '__main__':
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageNetV2Dataset(variant="matched-frequency", transform=preprocess)
    
    # FIX: Split data into Calibration (for gatekeepers) and Validation (for testing)
    calib_subset = torch.utils.data.Subset(dataset, range(0, 5000))
    val_subset = torch.utils.data.Subset(dataset, range(5000, 10000))
    
    # FIX: Add num_workers and pin_memory for faster data loading
    calib_loader = torch.utils.data.DataLoader(calib_subset, batch_size=32, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, num_workers=4, pin_memory=True)
    
    gates = train_gatekeepers(calib_loader)
    system = GatekeeperCascade(gates).to(device)
    
    print("\nCascade deployed. Evaluating on unseen validation data...")
    top1_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in val_loader: # Evaluate on val_loader!
            total_samples += labels.size(0)
            outputs = system(images.to(device))
            top1_correct += outputs.argmax(dim=1).eq(labels.to(device)).sum().item()

    print(f"\nOverall Accuracy: {top1_correct / total_samples:.4f}")
    system.print_stats(total_samples)