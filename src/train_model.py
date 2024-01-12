import timm
import torch
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn, optim
import hydra
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config.yaml")
def train(config):
    """Train the model on resnet50"""
    model_name = 'resnet50'

    hparams = config.train
    log.info(f"Hyperparameters: {hparams}")

# Load the ResNet model using timm
    model = timm.create_model(model_name, pretrained=False)  
    model.Dropout = nn.Dropout(hparams['dropout_rate'])
    root = hydra.core.hydra_config.HydraConfig.get().runtime.cwd
    subfolder = os.path.join(hparams['processed_dataset'], hparams['dataset_name'])

# Use os.path.join to join the paths
    data_folder = os.path.join(root, subfolder)
    print(data_folder)

    images = torch.load(os.path.join(data_folder, "fruit_training_images.pt"))
    labels = torch.load(os.path.join(data_folder, "fruit_training_labels.pt"))

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
    
    trainloader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=hparams["shuffle"])
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    optimizer.zero_grad()
    
    error = []
    steps = []
    count = 0
    for e in range(hparams["epochs"]):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector

            # TODO: Training pass
            
            log_probs = model(images)

            # Calculate the loss
            labels = labels.long()
            loss = criterion(log_probs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()
            running_loss += loss.item()
            print("Loss: ", loss.item())
        else:
            count += 1
            error.append(running_loss / len(trainloader))
            steps.append(count)
            print(f"Training loss: {running_loss/len(trainloader)}")
        
        print("Epoch number: ", e)
    
    plt.figure()
    plt.plot(steps, error)

    fig_path = "reports/figures/training_loss.png"
    plt.savefig(
        os.path.join(root, fig_path)
    )

    path = os.getcwd()
    save_path = os.path.join(root, "models/trained_model.pth")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(save_path)
    torch.save(model.state_dict(), save_path)

train()
