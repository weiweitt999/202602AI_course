import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


transform = transforms.Compose([
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)


test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)


print("Number of training images:", len(train_dataset))
print("Number of test images:", len(test_dataset))


image, label = train_dataset[0]

print("Image tensor shape:", image.shape)
print("Label index:", label)
print("Label name:", classes[label])


image_np = image.permute(1, 2, 0).numpy()

plt.imshow(image_np)
plt.title(f"Label: {classes[label]}")
plt.axis("off")
plt.show()