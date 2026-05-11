import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt


# =========================
# 1. 基本設定
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-6
num_epochs = 20

os.makedirs("results", exist_ok=True)

print("Using device:", device)


# =========================
# 2. CIFAR-10 transform
# =========================
# Linear probing 不需要 SimCLR 那麼強的 augmentation
# 訓練時用簡單 augmentation
# 測試時只做 ToTensor + Normalize

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

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
    transform=train_transform
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
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# =========================
# 4. 定義跟 SimCLR 一樣的模型
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
# 5. Linear Probe Model
# =========================

class LinearProbe(nn.Module):
    def __init__(self, backbone):
        super(LinearProbe, self).__init__()

        self.backbone = backbone

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 只訓練這一層
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)

        out = self.classifier(h)
        return out


# =========================
# 6. 載入 SimCLR 訓練好的 backbone
# =========================

simclr_model = SimCLRModel(projection_dim=128).to(device)
simclr_model.load_state_dict(
    torch.load("results/simclr_model_final.pth", map_location=device)
)

backbone = simclr_model.backbone

model = LinearProbe(backbone).to(device)


# =========================
# 7. Loss / Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)


# =========================
# 8. 評估函式
# =========================

def evaluate(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


# =========================
# 9. Linear Probing Training
# =========================

train_acc_history = []
test_acc_history = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)

    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Test Acc: {test_acc:.2f}%"
    )


# =========================
# 10. 儲存結果
# =========================

torch.save(model.state_dict(), "results/linear_probe_model.pth")

plt.figure()
plt.plot(range(1, num_epochs + 1), train_acc_history, marker="o", label="Train Acc")
plt.plot(range(1, num_epochs + 1), test_acc_history, marker="o", label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Linear Probing Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("results/linear_probe_accuracy.png", dpi=300)
plt.show()

print("Linear probing model saved as results/linear_probe_model.pth")
print("Accuracy curve saved as results/linear_probe_accuracy.png")
print(f"Final Linear Probing Test Accuracy: {test_acc_history[-1]:.2f}%")