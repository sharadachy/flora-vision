import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CustomCNN
import os

# ----------------- SETTINGS -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dir = "G:\Floravision8\dataset_split\test"  # Change to your test dataset folder
batch_size = 32
image_size = 128

# ----------------- TRANSFORMS -----------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----------------- DATASET -----------------
def safe_loader(path):
    from PIL import Image
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except:
        print(f"⚠ Skipping corrupted image: {path}")
        return None

test_dataset = datasets.ImageFolder(root=test_dir, transform=transform, loader=safe_loader)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = test_dataset.classes
print("Classes:", classes)

# ----------------- LOAD MODEL -----------------
model = CustomCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ----------------- EVALUATION -----------------
correct = 0
total = 0
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Record wrong predictions
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                misclassified.append({
                    "image_path": test_dataset.samples[i][0],
                    "predicted": classes[predicted[i].item()],
                    "actual": classes[labels[i].item()]
                })

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Total Misclassified Images: {len(misclassified)}")

# Optional: print first 10 misclassified images
for item in misclassified[:10]:
    print(f"Image: {item['image_path']}, Predicted: {item['predicted']}, Actual: {item['actual']}")

