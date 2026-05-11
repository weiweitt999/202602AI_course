import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt


# =========================
# 1. Basic Settings
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
k = 20

os.makedirs("results", exist_ok=True)

print("Using device:", device)


# =========================
# 2. CIFAR-10 Transform
# =========================
# kNN monitor 不用 random augmentation
# 因為我們要穩定抽 feature

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# =========================
# 3. Dataset / DataLoader
# =========================

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=test_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# =========================
# 4. Define SimCLR Model
# =========================

class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLRModel, self).__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


# =========================
# 5. Load SimCLR Final Model
# =========================

simclr_model = SimCLRModel(projection_dim=128).to(device)
simclr_model.load_state_dict(
    torch.load("results/simclr_model_final.pth", map_location=device)
)

backbone = simclr_model.backbone
backbone.eval()

print("Loaded model: results/simclr_model_final.pth")


# =========================
# 6. Extract Features
# =========================

def extract_features(backbone, data_loader):
    all_features = []
    all_labels = []

    backbone.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            features = backbone(images)          # [batch, 512]
            features = F.normalize(features, dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels


print("Extracting train features...")
train_features, train_labels = extract_features(backbone, train_loader)

print("Extracting test features...")
test_features, test_labels = extract_features(backbone, test_loader)

print("Train features shape:", train_features.shape)
print("Test features shape:", test_features.shape)


# =========================
# 7. kNN Monitor
# =========================

def knn_predict(train_features, train_labels, test_features, test_labels, k=20, chunk_size=500):
    correct = 0
    total = test_features.size(0)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)

        test_chunk = test_features[start:end].to(device)
        label_chunk = test_labels[start:end].to(device)

        # cosine similarity because features were normalized
        similarity = torch.matmul(test_chunk, train_features.T)

        # get top-k nearest neighbors
        _, indices = similarity.topk(k=k, dim=1)

        nearest_labels = train_labels[indices]   # [chunk, k]

        # majority vote
        predictions = []
        for i in range(nearest_labels.size(0)):
            vote = torch.bincount(nearest_labels[i], minlength=10)
            pred = vote.argmax()
            predictions.append(pred)

        predictions = torch.stack(predictions).to(device)

        correct += (predictions == label_chunk).sum().item()

        print(f"Processed {end}/{total} test images")

    accuracy = 100.0 * correct / total
    return accuracy


knn_acc = knn_predict(
    train_features=train_features,
    train_labels=train_labels,
    test_features=test_features,
    test_labels=test_labels,
    k=k,
    chunk_size=500
)

print(f"kNN Monitor Accuracy, k={k}: {knn_acc:.2f}%")


# =========================
# 8. Save Result and Figure
# =========================

with open("results/knn_monitor_result.txt", "w") as f:
    f.write(f"kNN Monitor Accuracy, k={k}: {knn_acc:.2f}%\n")

plt.figure()
plt.bar([f"k={k}"], [knn_acc])
plt.ylabel("Accuracy (%)")
plt.title("Final kNN Monitor Accuracy")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.savefig("results/knn_monitor_accuracy.png", dpi=300)
plt.show()

print("Saved: results/knn_monitor_result.txt")
print("Saved: results/knn_monitor_accuracy.png")