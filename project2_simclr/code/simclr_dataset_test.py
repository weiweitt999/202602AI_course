import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


# SimCLR augmentation
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


class SimCLRDataset(Dataset):
    def __init__(self, root="./data", train=True, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=False,
            transform=None
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # 對同一張圖做兩次 augmentation
        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2, label


# 建立 dataset
train_dataset = SimCLRDataset(
    root="./data",
    train=True,
    transform=simclr_transform
)

# 建立 dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)


# 測試抓一個 batch
view1_batch, view2_batch, labels = next(iter(train_loader))

print("view1 batch shape:", view1_batch.shape)
print("view2 batch shape:", view2_batch.shape)
print("labels shape:", labels.shape)
print("labels:", labels)