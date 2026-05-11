import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# SimCLR 常用的資料增強
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


# 為了顯示圖片，需要把 Normalize 還原
def denormalize(img):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    img = img.clone()

    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    img = img.clamp(0, 1)
    return img


# 注意：這裡先不放 transform
# 因為我們要拿 PIL image，然後自己做兩次 augmentation
dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=None
)


# 取出第 0 張原圖
image, label = dataset[0]


# 對同一張圖片做兩次不同 augmentation
view1 = simclr_transform(image)
view2 = simclr_transform(image)


# 還原成可以顯示的圖片
view1_show = denormalize(view1).permute(1, 2, 0).numpy()
view2_show = denormalize(view2).permute(1, 2, 0).numpy()


plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(view1_show)
plt.title("View 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(view2_show)
plt.title("View 2")
plt.axis("off")

plt.suptitle(f"Original Label: {classes[label]}")
plt.tight_layout()
plt.show()