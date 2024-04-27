import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.dataset import DominoDataset
from model.network import DominoCNN
import os

def load_model(path):
    """Load the trained model from the specified path and set it to evaluation mode."""
    model = DominoCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def validate_model(model, loader):
    """Calculate and print the accuracy of the model on the validation set and display detailed prediction results."""
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Output detailed prediction results
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            for idx, (prob, pred) in enumerate(zip(probabilities, predicted)):
                print(f'Image {idx}: Predicted Class = {pred.item()}, Probabilities = {prob.numpy()}')

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    # Set the path to the validation dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = DominoDataset(csv_file='data/validation_labels.csv', root_dir='data/processed/validation', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Load the model from the specified path
    model_path = 'model/domino_cnn.pth'
    model = load_model(model_path)

    # Validate the model using the loaded data and print detailed outputs
    validate_model(model, val_loader)
