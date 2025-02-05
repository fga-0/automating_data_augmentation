import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import os

# Define transformations (normalization, conversion to tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
inverse_transform = transforms.ToPILImage()

# Load full CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transform)

# Select only 1% of the dataset
num_train_samples = 500  # 1% of 50,000
num_test_samples = 100    # 1% of 10,000

# Randomly select indices
train_indices = np.random.choice(len(trainset), num_train_samples, replace=False)
test_indices = np.random.choice(len(testset), num_test_samples, replace=False)

# Create subset datasets
train_subset = Subset(trainset, train_indices)
test_subset = Subset(testset, test_indices)

# Create DataLoaders
# train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

# Verify dataset size
# print(f"Training samples: {len(train_loader.dataset)}")
# print(f"Test samples: {len(test_loader.dataset)}")

# Create directory to save the reduced dataset
save_dir = "./cifar10_reduced"
train_dir = os.path.join(save_dir, "train")
test_dir = os.path.join(save_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Save selected images as PNG files
for i, idx in enumerate(train_indices):
    img, label = trainset[idx]
    img = inverse_transform(img)
    img.save(os.path.join(train_dir, f"{i}_{label}.png"))

for i, idx in enumerate(test_indices):
    img, label = testset[idx]
    img = inverse_transform(img)
    img.save(os.path.join(test_dir, f"{i}_{label}.png"))

print(f"Saved {num_train_samples} training images and {num_test_samples} test images in {save_dir}")