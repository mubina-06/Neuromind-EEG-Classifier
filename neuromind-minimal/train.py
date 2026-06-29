"""Training pipeline"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from model import build_model, save_model

class EEGDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        classes = ["focused", "relaxed", "stressed"]
        
        for i, cls in enumerate(classes):
            cls_dir = Path(data_dir) / cls
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((str(img_path), i))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(split="train"):
    """Data transforms"""
    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def train_model(arch="resnet18", epochs=10):
    """Train CNN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    dataset = EEGDataset("data", get_transforms("train"))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # Model
    model = build_model(arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.3f}")
        
        if acc > best_acc:
            best_acc = acc
            save_model(model, arch)
    
    return model