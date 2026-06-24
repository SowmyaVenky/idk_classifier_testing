import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from imagenetv2_pytorch import ImageNetV2Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ThreeStageResNetCascade(nn.Module):
    def __init__(self, t1=0.5, t2=0.85):
        super(ThreeStageResNetCascade, self).__init__()
        # Confidence thresholds
        self.t1 = t1  # Stage 1 -> Stage 2 threshold
        self.t2 = t2  # Stage 2 -> Stage 3 threshold
        
        # Initialize the three models
        self.stage1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
        self.stage2 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device).eval()
        self.stage3 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device).eval()
        
        # Track routing stats
        self.total_images = 0
        self.s2_count = 0
        self.s3_count = 0

    def forward(self, x):
        batch_size = x.size(0)
        self.total_images += batch_size
        
        with torch.no_grad():
            # --- STAGE 1: ResNet18 ---
            out1 = self.stage1(x)
            probs1 = F.softmax(out1, dim=1)
            max_p1, _ = torch.max(probs1, dim=1)
            
            # Mask for images failing Stage 1 threshold
            to_s2_mask = max_p1 < self.t1
            final_outputs = out1.clone()
            
            if not to_s2_mask.any():
                return final_outputs
                
            # --- STAGE 2: ResNet34 ---
            s2_images = x[to_s2_mask]
            self.s2_count += s2_images.size(0)
            
            out2 = self.stage2(s2_images)
            probs2 = F.softmax(out2, dim=1)
            max_p2, _ = torch.max(probs2, dim=1)
            
            # Sub-mask: identify which Stage 2 images must go to Stage 3
            to_s3_sub_mask = max_p2 < self.t2
            
            # Store Stage 2 results into final outputs
            final_outputs[to_s2_mask] = out2
            
            if not to_s3_sub_mask.any():
                return final_outputs
                
            # --- STAGE 3: ResNet152 ---
            # Map sub-mask indices back to the original batch scale
            to_s3_indices = torch.where(to_s2_mask)[0][to_s3_sub_mask]
            s3_images = x[to_s3_indices]
            self.s3_count += s3_images.size(0)
            
            out3 = self.stage3(s3_images)
            
            # Store Stage 3 results into final outputs
            final_outputs[to_s3_indices] = out3
            
            return final_outputs

if __name__ == '__main__':
    # Instantiate cascade model
    # t1=0.5 (ResNet18 handles easy), t2=0.85 (ResNet34 handles medium, ResNet152 gets hardest)
    cascade = ThreeStageResNetCascade(t1=0.5, t2=0.85)

    # Standard Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load ImageNet-V2 Dataset
    dataset = ImageNetV2Dataset(variant="matched-frequency", transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # Evaluation Loop
    top1_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = cascade(images)
            
            _, preds = outputs.topk(1, dim=1)
            total_samples += labels.size(0)
            top1_correct += preds.eq(labels.view_as(preds)).sum().item()

    # Calculations for distribution metrics
    s1_only = cascade.total_images - cascade.s2_count
    s2_only = cascade.s2_count - cascade.s3_count
    s3_only = cascade.s3_count

    print(f"Top-1 Accuracy: {top1_correct / total_samples:.4f}")
    print(f"Processed by ResNet18 only:  {s1_only / cascade.total_images * 100:.2f}%")
    print(f"Processed by ResNet34:       {s2_only / cascade.total_images * 100:.2f}%")
    print(f"Processed by ResNet152:      {s3_only / cascade.total_images * 100:.2f}%")
