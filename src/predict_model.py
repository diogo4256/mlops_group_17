import torch
import timm
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image

def load_image(image_path):
    """Load a single image."""
    image = Image.open(image_path)
    return image

def predict(data_folder, mode='batch'):
    """Make predictions with the trained model"""
    model_name = 'resnet50'

    # Load the model
    model = timm.create_model(model_name, pretrained=False)
    root = os.getcwd()
    save_path = os.path.join(root, "models/trained_model.pth")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if mode == 'single':
        # Load a single image
        image_path = os.path.join(root, data_folder)
        images = [load_image(image_path)]
        labels = None
    else:
        # Load a batch of images
        images = torch.load(os.path.join(data_folder, "fruit_training_images.pt"))
        labels = torch.load(os.path.join(data_folder, "fruit_training_labels.pt"))

    # Create a DataLoader
    dataset = CustomDataset(images, labels, transform=preprocess)
    batch_size = 1 if mode == 'single' else images.shape[0] // 10
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Make predictions and calculate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            if mode == 'single':
                images = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                print('Predicted label for the image:', predicted.item())
            else:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    if mode != 'single':
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        print(predicted)
        return(100 * correct / total)
    
    else:
        return(predicted.item())


if __name__ == "__main__":
    predict()