import timm
import torch
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn, optim
import hydra
import logging
import wandb
import omegaconf

log = logging.getLogger(__name__)
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
    
def plot_training_loss(error, steps, root):
    log.info("Plotting training loss...")
    plt.figure()
    plt.plot(steps, error)
    log.info("Plot created.")

    fig_path = "reports/figures/training_loss.png"
    log.info(f"Saving plot to {fig_path}...")
    plt.savefig(
        os.path.join(root, fig_path)
    )
    log.info("Plot saved.")

def load_data(root, hparams):
    log.info("Loading data...")
    subfolder = os.path.join(hparams['processed_dataset'], hparams['dataset_name'])
    data_folder = os.path.join(root, subfolder)
    images = torch.load(os.path.join(data_folder, "fruit_training_images.pt"))
    labels = torch.load(os.path.join(data_folder, "fruit_training_labels.pt"))
    dataset = CustomDataset(images, labels)
    trainloader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=hparams["shuffle"])
    log.info("Data loaded.")
    return trainloader

def setup_model_and_optimizer(hparams):
    log.info("Setting up model and optimizer...")
    model = timm.create_model('resnet50', pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    log.info("Model and optimizer set up.")
    return model, criterion, optimizer

def train_one_epoch(model, criterion, optimizer, trainloader):
    log.info("Training for one epoch...")
    running_loss = 0
    for images, labels in trainloader:
        wandb.log({"example_image": wandb.Image(images[0])})
        log_probs = model(images)
        labels = labels.long()
        loss = criterion(log_probs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({"loss": loss.item()})
        
    log.info("Training for one epoch completed.")
    # Log the training loss
    wandb.log({"training_loss": running_loss/len(trainloader)}) 
    return running_loss

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def train(config):
    log.info("Starting training...")
    
    wandbconfig = config.wandb
    wandb.config = omegaconf.OmegaConf.to_container(
        wandbconfig, resolve=True, throw_on_missing=True
    )

    # Get the API key from the environment variables
    wandb_api_key = os.environ['WANDB_API_KEY']

    # Use the API key when initializing wandb
    wandb.login(key=wandb_api_key)
    wandb.init(project=wandbconfig["project"], entity=wandbconfig["entity"])
    
    hparams = config.train
    root = hydra.core.hydra_config.HydraConfig.get().runtime.cwd
    trainloader = load_data(root, hparams)
    model, criterion, optimizer = setup_model_and_optimizer(hparams)

    error = []
    steps = []
    for e in range(hparams["epochs"]):
        running_loss = train_one_epoch(model, criterion, optimizer, trainloader)
        error.append(running_loss / len(trainloader))
        steps.append(e+1)
        log.info(f"Epoch {e} completed.")

    plot_training_loss(error, steps, root)

    save_path = os.path.join(root, hparams["model_savepath"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info("Training completed.")

if __name__ == "__main__":
    train()
