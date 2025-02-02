import matplotlib.pyplot as plt
import numpy as np

def show_images(dataset, num=5):
    fig, axes = plt.subplots(1, num, figsize=(10, 3))
    random_idx = [np.random.randint(0, len(dataset)) for _ in range(num)]
    for i in range(num):
        img, label = dataset[random_idx[i]]
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.show()