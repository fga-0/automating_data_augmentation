import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset


# Datasets
def get_subset(trainset,
               testset, 
               percentage) : 
    # Select only 1% of the dataset
    num_train_samples = int(percentage*len(trainset))
    num_test_samples = int(percentage*len(testset))    

    # Randomly select indices
    train_indices = np.random.choice(len(trainset), num_train_samples, replace=False)
    test_indices = np.random.choice(len(testset), num_test_samples, replace=False)

    # Create subset datasets
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)

    # Verify dataset size
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")
    return train_subset, test_subset

# Plots
def plot_images(dataset, 
                n_images
                ) : 
    """
    This function plots n_images images selected randomly within the dataset.
    """
    size = n_images
    image_idx = np.random.randint(low=0, high=len(dataset), size=size)
    fig, ax = plt.subplots(nrows=1, ncols=size, figsize=(10, 3))
    for i in range(size) : 
        idx = image_idx[i]
        raw_image = dataset[idx][0] 
        ax[i].imshow(np.transpose(raw_image, (1,2,0)))
        ax[i].axis("off")        
    plt.show()

def plot_before_after_augmentation(dataset, 
                                   transformation : transforms
                                       ) : 
    size = 5
    image_idx = np.random.randint(low=0, high=len(dataset), size=size)
    fig, ax = plt.subplots(nrows=2, ncols=size, figsize=(10, 3))
    for i in range(size) : 
        idx = image_idx[i]
        raw_image = dataset[idx][0] 
        # Display raw image
        ax[0][i].imshow(np.transpose(raw_image, (1,2,0)))
        ax[0][i].axis("off")
        # Display transformed image
        transformed_img = transformation(raw_image)
        ax[1][i].imshow(np.transpose(transformed_img, (1,2,0)))
        ax[1][i].axis("off")
        
    plt.show()
    
def display_one_image_per_class() : 
    return

def train_WideResNet(
    num_epochs : int, 
    batch_size,
    model, 
    trainloader, 
    optimizer, 
    criterion,
    device, 
    scheduler, 
    ) : 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        losses = []
        accuracies = []
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{running_loss / (total / batch_size):.4f}", acc=f"{100 * correct / total:.2f}%")
            losses.append(loss)
        scheduler.step()
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}: Loss={running_loss/len(trainloader):.4f}, Accuracy={accuracy:.2f}%")

    print("Training complete!")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].plot(range(num_epochs), losses)
    axes[0].title.set_text("Training loss")
    axes[1].plot(range(num_epochs), accuracies)
    axes[1].title.set_text("Training accuracy")
    plt.show()
    return model


def evaluate(model, test_loader, device):
    model.eval()  # Mode évaluation (désactive dropout, batchnorm)
    correct = 0
    total = 0

    with torch.no_grad():  # Pas de calcul de gradient
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    error_rate = 1 - accuracy
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Taux d'erreur: {error_rate * 100:.2f}%")
    return accuracy, error_rate
