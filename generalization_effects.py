import torch
import random
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

class AugmentedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def uncertainty_based_sampling(
    batch, # B data points (x1,y1), ..., (xB, yB)
    F_transformations, # K transformations F1, ..., FK
    G_default_transformations, # G default transformations 
    model, 
    loss_function,
    L, # the number of composition steps
    C, # number of augmented data per input data
    S, # number of selected data points used for training
    device
): 
    model.eval()
    images, labels = zip(*batch)
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    
    selected_samples = []  # Store selected high-uncertainty samples
    selected_labels = []   # Store corresponding labels

    with torch.no_grad():
        for i in range(len(images)):  # Loop over each image in batch
            augmented_samples = []
            loss_values = []
            
            for _ in range(C):  # Create C augmented versions
                transformed_img = images[i].cpu()

                # Apply L random transformations
                transforms_list = random.sample(F_transformations, L)
                transform_pipeline = transforms.Compose(transforms_list)                
                transformed_img = transform_pipeline(transformed_img)
                                                
                transformed_img = G_default_transformations(transformed_img)
                
                transformed_img = transformed_img.to(device) 
                
                transformed_img = transformed_img.unsqueeze(0)  # Add batch dim
                
                # Compute loss
                output = model(transformed_img)
                loss = loss_function(output, labels[i].unsqueeze(0))
                
                augmented_samples.append(transformed_img.squeeze(0).cpu())  # Remove batch dim before storing
                loss_values.append(loss.item())

            # Select the top S samples with highest loss
            top_s_indices = np.argsort(-np.array(loss_values))[:S]
            selected_samples.extend([augmented_samples[idx] for idx in top_s_indices])
            selected_labels.extend([labels[i].item()] * S)  # Store corresponding labels

    # Return dataset in correct format
    return AugmentedDataset(selected_samples, selected_labels, transform=None)
