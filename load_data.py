from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
"""Скачування даних за допомогою DataLoader.
Data - MNIST
Потім її конвертуємо у тензори, для подальшої роботи у ПайТорч
Її батчимо її для подальшого тренування
"""

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
