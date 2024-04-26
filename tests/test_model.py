import sys
import os
import torch
from torch.utils.data import DataLoader

# Adjust the path to include the model directory
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.network import DominoCNN  # Assuming the model definition is in network.py
from model.dataset import DominoDataset  # Assuming the dataset definition is in dataset.py

def load_model(path):
    """ Load the trained model from a file """
    model = DominoCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on test images: {accuracy:.2f}%')

if __name__ == "__main__":
    test_dataset = DominoDataset(csv_file='data/labels.csv', root_dir='data/processed')  # Ensure you have test data prepared
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model_path = 'model/domino_cnn.pth'  # Update with the actual path where your model is saved
    model = load_model(model_path)
    test_model(model, test_loader)
