import timm
import torch
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn, optim
import hydra
import logging
import wandb

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def train(config):
    """Train the model on resnet50"""
    model_name = "resnet50"

    # Get the hyperparameters from the config file
    hparams = config.train
    log.info(f"Hyperparameters: {hparams}")

    log.info(f"Loading model: {model_name}")

    model = timm.create_model(model_name, pretrained=False)
    root = hydra.core.hydra_config.HydraConfig.get().runtime.cwd
    subfolder = os.path.join(hparams["processed_dataset"], hparams["dataset_name"])

    # Use os.path.join to join the paths
    data_folder = os.path.join(root, subfolder)
    log.info(f"Data folder: {data_folder}")

    log.info("Loading images and labels...")
    images = torch.load(os.path.join(data_folder, "fruit_training_images.pt"))
    labels = torch.load(os.path.join(data_folder, "fruit_training_labels.pt"))
    log.info("Images and labels loaded.")

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

    log.info("Creating DataLoader...")
    dataset = CustomDataset(images, labels)

    trainloader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=hparams["shuffle"])
    log.info("DataLoader created.")

    log.info("Setting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    optimizer.zero_grad()
    log.info("Loss function and optimizer set up.")

    error = []
    steps = []
    count = 0
    log.info("Starting training...")
    for e in range(hparams["epochs"]):
        running_loss = 0
        for images, labels in trainloader:
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
            # Log the loss
            log.info({"loss": loss.item()})
        else:
            count += 1
            error.append(running_loss / len(trainloader))
            steps.append(count)

            # Log the training loss
            log.info({"training_loss": running_loss / len(trainloader)})

        log.info(f"Epoch {e}")

    log.info("Training complete.")
    log.info("Plotting training loss...")
    plt.figure()
    plt.plot(steps, error)
    log.info("Plot created.")

    fig_path = "reports/figures/training_loss.png"
    log.info(f"Saving plot to {fig_path}...")
    plt.savefig(os.path.join(root, fig_path))
    log.info("Plot saved.")

    save_path = os.path.join(root, "models/trained_model.pth")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    log.info(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    log.info("Model saved.")


train()
