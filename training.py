import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from model import CustomCNN
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pillow_avif
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- PATHS -----------------
train_dir = "E:/FINAL_PROJECT/dataset_split/train"
val_dir = "E:/FINAL_PROJECT/dataset_split/val"

# ----------------- HYPERPARAMETERS -----------------
batch_size = 32
epochs = 30
learning_rate = 0.001
image_size = 128
patience = 5

print("Training started...")

# ----------------- DEVICE -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- TRANSFORMS -----------------
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----------------- SAFE LOADER -----------------
def safe_loader(path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as e:
        print(f"Skipping corrupted image: {path}")
        return None

# ----------------- DATASETS -----------------
train_data = datasets.ImageFolder(train_dir, transform=train_transform, loader=safe_loader)
val_data = datasets.ImageFolder(val_dir, transform=val_transform, loader=safe_loader)

# 🔥 SAVE CLASS MAPPING (VERY IMPORTANT)
torch.save(train_data.class_to_idx, "class_to_idx.pth")
print("Saved class_to_idx.pth")

# Handle imbalance
targets = [label for _, label in train_data.samples]
class_counts = torch.bincount(torch.tensor(targets))
class_weights = 1. / class_counts.float()
sample_weights = [class_weights[label] for label in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_data, batch_size=batch_size)

classes = train_data.classes
print("Classes:", classes)

# ----------------- MODEL -----------------
model = CustomCNN(num_classes=len(classes)).to(device)

# ----------------- OPTIMIZER & LOSS -----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_acc = 0.0
patience_counter = 0

# ----------------- TRAINING LOOP -----------------
for epoch in range(epochs):

    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    scheduler.step()

    # ----------------- VALIDATION -----------------
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # ----------------- BEST MODEL SAVE -----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("New best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("Training completed.")
print("Best Validation Accuracy:", best_val_acc)
