import sys
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Ensure that the root directory is in the path for importing DominoDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.dataset import DominoDataset

def show_images(images, labels, num_images=4):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].numpy().transpose(1, 2, 0))  # Convert from Tensor image
        ax.title.set_text('Label: {}'.format(labels[i]))
        ax.axis('off')
    plt.show()

def test_data_loader():
    # Initialize the dataset and dataloader
    dataset = DominoDataset(csv_file='data/labels.csv', root_dir='data/processed')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test the loader
    for images, labels in dataloader:
        show_images(images, labels)
        break  # Only show the first batch

if __name__ == "__main__":
    test_data_loader()
