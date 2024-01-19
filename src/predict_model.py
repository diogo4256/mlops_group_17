import torch
import timm
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pickle
from google.cloud import storage


def download_file_from_gcs(bucket_name, blob_name, destination_path, file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    final_path = os.path.join(destination_path, file_name)
    blob.download_to_filename(final_path)
    return final_path


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


def load_folder_names_and_labels(dict_path):
    """Load the dictionary of number to labels from the pickle file."""
    with open(dict_path, "rb") as f:
        folder_names_and_labels = pickle.load(f)
    return folder_names_and_labels


def predict(data_folder, mode="single"):
    """Make predictions with the trained model"""
    bucket_name = "fruit_bucket_mlops"
    blob_name = "models/trained_model.pth"
    destination_path = "/models"
    file_name = "trained_model2.pth"

    model_path = download_file_from_gcs(bucket_name, blob_name, destination_path, file_name)

    bucket_name = "fruit_bucket_mlops"
    blob_name = "data/processed/folder_names_and_labels.pkl"
    destination_path = "/models"
    file_name = "folder_names_and_labels.pkl"

    dict_path = download_file_from_gcs(bucket_name, blob_name, destination_path, file_name)

    model_name = "resnet50"

    # Load the model
    model = timm.create_model(model_name, pretrained=False)
    root = os.getcwd()
    # save_path = "gs://fruit_bucket_mlops/models/trained_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    """
    def download_file_from_gcs(bucket_name, blob_name, destination_path, file_name):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        final_path = os.path.join(destination_path, file_name)
        blob.download_to_filename(final_path)
        return final_path
    """
    if mode == "single":
        # Load a single image
        bucket_name = "fruit_bucket_mlops"
        blob_name = data_folder
        destination_path = "/models"
        file_name = "image.jpg"

        image_path = download_file_from_gcs(bucket_name, blob_name, destination_path, file_name)
        #image_path = os.path.join(root, data_folder)
        images = [load_image(image_path)]
        labels = None
    else:
        # Load a batch of images
        images = torch.load(os.path.join(data_folder, "fruit_training_images.pt"))
        labels = torch.load(os.path.join(data_folder, "fruit_training_labels.pt"))

    # Create a DataLoader
    dataset = CustomDataset(images, labels, transform=preprocess)
    batch_size = 1 if mode == "single" else images.shape[0] // 10
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Make predictions and calculate accuracy
    correct = 0
    total = 0
    # dictionary of number to labels
    label_dict = load_folder_names_and_labels(dict_path)
    with torch.no_grad():
        for data in dataloader:
            if mode == "single":
                images = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = label_dict[predicted.item()]  # predicted.item() is the number
                print("Predicted label for the image:", predicted.item())
                print("Predicted label for the image:", predicted_label)
            else:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    if mode != "single":
        print("Accuracy of the model on the test images: {} %".format(100 * correct / total))
        predicted_labels = [label_dict[label.item()] for label in predicted]
        print(predicted)
        print(predicted_labels)
        return 100 * correct / total

    else:
        predicted_label = label_dict[predicted.item()]
        return predicted_label


if __name__ == "__main__":
    predict()
