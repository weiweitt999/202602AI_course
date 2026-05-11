import torch
import torch.nn as nn
import torchvision.models as models


class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLRModel, self).__init__()

        # ResNet-18，從零開始訓練，不使用 pretrained weights
        self.backbone = models.resnet18(weights=None)

        # CIFAR-10 是 32x32 小圖
        # 原本 ResNet-18 conv1 是 7x7 stride=2，不適合 CIFAR-10
        # 改成 3x3 stride=1 padding=1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # 拿掉 maxpool，避免圖片太快變小
        self.backbone.maxpool = nn.Identity()

        # 拿掉最後分類層，讓 backbone 輸出 512 維 feature
        self.backbone.fc = nn.Identity()

        # SimCLR projector head: 512 -> 512 -> 128
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.backbone(x)      # h: [batch_size, 512]
        z = self.projector(h)     # z: [batch_size, 128]
        return h, z


if __name__ == "__main__":
    model = SimCLRModel(projection_dim=128)

    # 假資料：假裝有 4 張 CIFAR-10 圖片
    x = torch.randn(4, 3, 32, 32)

    h, z = model(x)

    print("Input x shape:", x.shape)
    print("Backbone output h shape:", h.shape)
    print("Projector output z shape:", z.shape)