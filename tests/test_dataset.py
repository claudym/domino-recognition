import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.dataset import DominoDataset

def test_domino_dataset():
    dataset = DominoDataset(csv_file='data/labels.csv', root_dir='data/processed')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
        print(f'Batch of labels: {labels}')  # Output labels for verification
        break  # Only process one batch for testing

if __name__ == '__main__':
    test_domino_dataset()
