import sys
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

# Adjust the path to include the root of the project
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.dataset import DominoDataset  # Import the dataset class
from model.network import DominoCNN     # Import the model class

def train_model():
    dataset = DominoDataset(csv_file='data/labels.csv', root_dir='data/processed')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = DominoCNN()  # Instantiate the model
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()  # Define the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define the optimizer

    num_epochs = 30
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # Save the model
    torch.save(model.state_dict(), 'model/domino_cnn.pth')
    print("Model saved successfully.")

if __name__ == '__main__':
    train_model()
