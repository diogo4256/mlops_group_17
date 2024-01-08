import timm
import torch
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn, optim

def train():

    model_name = 'resnet50'

# Load the ResNet model using timm
    model = timm.create_model(model_name, pretrained=False)  
    path = os.getcwd()
    subfolder = "data/processed"

# Use os.path.join to join the paths
    data_folder = os.path.join(path, subfolder)
    print(data_folder)
    #print(path)
    #print(os.path.join(path, "/data/processed"))
    #data_folder = os.path.join(path, "/data/processed")
    #print(data_folder)

    images = torch.load(os.path.join(data_folder, "train_images.pt"))
    labels = torch.load(os.path.join(data_folder, "train_target.pt"))

    class CustomDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            image = self.images[index]
            label = self.labels[index]

            return image, label

    dataset = CustomDataset(images, labels)
    batch_size = 64
    shuffle = True
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    optimizer.zero_grad()
    epochs = 10
    error = []
    steps = []
    count = 0
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            #images = images.view(images.shape[0], -1)
            print(images.shape)
            #images = images.view(64, 3, 28, 28)
            # TODO: Training pass
            log_probs = model(images)

            # Calculate the loss
            loss = criterion(log_probs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()
            running_loss += loss.item()
        else:
            count += 1
            error.append(running_loss / len(trainloader))
            steps.append(count)
            print(f"Training loss: {running_loss/len(trainloader)}")
    plt.figure()
    plt.plot(steps, error)

    fig_path = "reports/figures/training_loss.png"
    plt.savefig(
        os.path.join(path, fig_path)
    )

    path = os.getcwd()
    save_path = os.path.join(path, "models/trained_model.pth")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(save_path)
    torch.save(model.state_dict(), save_path)


train()
