
import torch
import torch.nn as nn
from torchvision import transforms


class BreakerCropClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(BreakerCropClassifier, self).__init__()
        # Simple CNN architecture for edge/texture classification
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 64 -> 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32 -> 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.classes = ["MCB", "RCD", "RCD_SI", "MAINBREAKER", "OVERSURGE", "OTHER"]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_image(self, pil_image, device="cpu"):
        """Runs inference on a single PIL image crop."""
        self.eval()
        tensor = self.transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        return self.classes[pred_idx.item()], float(conf.item())
