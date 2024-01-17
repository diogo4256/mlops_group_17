import torch
import timm
import os
from torch.utils.data import Dataset, DataLoader

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


def predict(data_folder):
    """Make predictions with the trained model"""
    model_name = 'resnet50'

    # Load the model
    model = timm.create_model(model_name, pretrained=False)
    root = os.getcwd()
    save_path = os.path.join(root, "models/trained_model.pth")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # Load the images
    images = torch.load(os.path.join(data_folder, "fruit_training_images.pt"))
    labels = torch.load(os.path.join(data_folder, "fruit_training_labels.pt"))

    # Create a DataLoader
    dataset = CustomDataset(images, labels)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Make predictions and calculate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return(100 * correct / total)


if __name__ == "__main__":
    predict("/Users/pawel/Desktop/ML_OPS_PROJ/mlops_group_17/data/processed/Apple Pink Lady")