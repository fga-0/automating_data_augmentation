import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models import wide_resnet50_2
import torch.nn as nn


class EarlyStopping: 
    def __init__(self, patience, min_delta_loss, min_delta_accuracy) :
        self.patience = patience
        self.min_delta_loss = min_delta_loss
        self.min_delta_accuracy = min_delta_accuracy
        self.best_loss = float('inf')
        # self.early_stop = False
        self.counter = 0
        # self.best_model_state = None
        self.best_accuracy = float('inf')
    def __call__(self, val_loss, val_accuracy) : 
        if self.best_loss - val_loss < self.min_delta_loss or abs(self.best_accuracy - val_accuracy) < self.min_delta_accuracy: 
            self.counter += 1
        else : 
            self.best_loss = val_loss
            self.counter = 0
        
        # if self.best_loss - val_loss > self.min_delta_loss or self.best_accuracy - val_accuracy > self.min_delta_accuracy: 
        #     self.best_loss = val_loss
        #     self.counter = 0
        # else : 
        #     self.counter += 1
        return self.counter >= self.patience
            
        

# Datasets

def get_trainloader_valloader(
    dataset, # the dataset to split in trainin ang validation sets
    split : float, # specifies the splitting between training and validation sets
    batch_size : int,
    ) : 
    
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)    
    print(f"Training set length : {len(trainset)}.")
    print(f"Validation set length : {len(valset)}.")
    return train_loader, val_loader
    
def get_subset(trainset,
               testset, 
               percentage) : 
    # Select only 1% of the dataset
    num_train_samples = int(percentage*len(trainset))
    num_test_samples = int(percentage*len(testset)*3)    

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
    

def plot_one_instance_per_class(dataset) : 
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 3))
    for label in range(10) : 
        if label<=4 : 
            row = 0
        else : 
            row = 1
        for i in range(len(dataset)) : 
            if dataset[i][1] == label : 
                ax[row][label%5].imshow(np.transpose(dataset[i][0], (1,2,0)))
                ax[row][label%5].title.set_text(f"{labels[label]}")
                ax[row][label%5].axis("off")
                break
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


def mix_images(image1, image2, alpha):
    return alpha * image1 + (1 - alpha) * image2


def demo_mixup(image1, image2, alpha):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.transpose(image1, (1, 2, 0)))
    ax[0].set_title('Image 1')
    ax[1].imshow(np.transpose(image2, (1, 2, 0)))
    ax[1].set_title('Image 2')
    ax[2].imshow(np.transpose(mix_images(image1, image2, alpha), (1, 2, 0)))
    ax[2].set_title(f'Mixup (alpha={alpha})')
    plt.show()

# Wide Residual Network

def load_Wide_Residual_Network(device : str, path : str) : 
    wrn = wide_resnet50_2()
    # Freeze all model parameters except for the final layer:
    for param in wrn.parameters():
        param.requires_grad = False
    # Get the number of input features for the original last layer:
    num_feature = wrn.fc.in_features
    # Replace the final classification layer to match your dataset:
    wrn.fc = nn.Linear(num_feature, 10)
    # View the structure of the new final layer :
    print(wrn.fc)
    # Move the model to the GPU for accelerated training:
    wrn = wrn.to(device)
    wrn.load_state_dict(torch.load(path, weights_only=False, map_location=torch.device('cpu')))
    return wrn

def train_WideResNet(
    num_epochs : int, 
    batch_size,
    model, 
    trainloader, 
    valloader,
    optimizer, 
    criterion,
    device, 
    scheduler, 
    patience, 
    min_delta_loss,
    min_delta_accuracy
    ) : 
    
    training_losses = []
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience, min_delta_loss=min_delta_loss, min_delta_accuracy=min_delta_accuracy)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

            pbar.set_postfix(loss=f"{running_loss / (total_train / batch_size):.4f}", acc=f"{100 * correct_train / total_train:.2f}%")
        
        avg_loss = running_loss / len(trainloader)
        training_losses.append(avg_loss)
        accuracy = 100 * correct_train / total_train
        train_accuracies.append(accuracy)
        
        # Validation step 
        model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad() : 
            for inputs, labels in valloader : 
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calcul de l'accuracy sur la validation
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)
                
        val_loss /= len(valloader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct_val / total_val  # Calcul de l'accuracy
        val_accuracies.append(val_accuracy)  # Stockage pour analyse
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.2f}%")
        scheduler.step()
        
        if early_stopping(val_loss, val_accuracy) :                 
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    print("Training complete!")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].plot(range(epoch+1), torch.tensor(training_losses).cpu(), label="Training loss")
    axes[0].plot(range(epoch+1), torch.tensor(val_losses).cpu(), label="Validation loss")
    axes[0].title.set_text("Losses")
    axes[0].set_xlabel(xlabel="Epochs")
    axes[0].legend(loc="upper right")
    axes[1].plot(range(epoch+1), torch.tensor(train_accuracies).cpu(), label="Training accuracies")
    axes[1].plot(range(epoch+1), torch.tensor(val_accuracies).cpu(), label="Validation accuracies")
    axes[1].title.set_text("Accuracies")
    axes[1].set_xlabel(xlabel="Epochs")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return model


def evaluate(model, test_loader, device):
    model.eval()  # Mode évaluation (désactive dropout, batchnorm)
    correct = 0
    total = 0

    with torch.no_grad():  # Pas de calcul de gradient
        for images, labels in tqdm(test_loader):
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
