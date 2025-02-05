import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm


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

        scheduler.step()
        print(f"Epoch {epoch+1}: Loss={running_loss/len(trainloader):.4f}, Accuracy={100 * correct / total:.2f}%")

    print("Training complete!")
