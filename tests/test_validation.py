import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.dataset import DominoDataset
from model.network import DominoCNN
import os

def load_model(path):
    model = DominoCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def validate_model(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    # Set the path to the validation dataset
    val_dataset = DominoDataset(csv_file='data/validation_labels.csv', root_dir='data/processed/validation',
                                transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Load the model
    model_path = 'model/domino_cnn.pth'
    model = load_model(model_path)

    # Validate the model
    validate_model(model, val_loader)
