import torch
import random
import torchvision.transforms as transforms
import numpy as np

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
    ) : 

    model.eval()
    images, labels = zip(*batch)
    images, labels = images.to(device), labels.to(device)
    
    selected_samples = []  # Store selected high-uncertainty samples
    
    with torch.no_grad():
        for i in range(len(images)):  # Loop over each image in batch
            augmented_samples = []
            loss_values = []
            
            for _ in range(C):  # Create C augmented versions
                transformed_img = images[i].cpu()

                # Apply L random transformations
                transforms_list = random.sample(F_transformations, L)
                transform_pipeline = transforms.Compose(transforms_list + G_default_transformations)
                
                transformed_img = transform_pipeline(transformed_img)
                transformed_img = transformed_img.to(device).unsqueeze(0)  # Add batch dim
                
                # Compute loss
                output = model(transformed_img)
                loss = loss_function(output, labels[i].unsqueeze(0))
                
                augmented_samples.append(transformed_img)
                loss_values.append(loss.item())

            # Select the top S samples with highest loss
            top_s_indices = np.argsort(-np.array(loss_values))[:S]
            selected_samples.extend([augmented_samples[idx] for idx in top_s_indices])

    return selected_samples  # Return S most uncertain augmented samples