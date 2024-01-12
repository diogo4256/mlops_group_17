import os
import torch
import torchvision.transforms as transforms
import zipfile
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
import hydra
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="config.yaml")
def main(config):
    
    hparams = config.make_data
    log.info(f"Hyperparameters: {hparams}")
    root = hydra.core.hydra_config.HydraConfig.get().runtime.cwd
    
    log.info("Starting data retrieval from API...")
    retrieve_from_api(os.path.join(root, hparams["path_extract"]), hparams["kaggle_dataset"])
    log.info("Data retrieval complete.")
    
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0,), (1,))])
    
    log.info("Loading and transforming images...")
    training_images = images_to_tensor(os.path.join(root, hparams["path_extract"], hparams["raw_dataset"]), transform=transform)
    training_labels = labels_to_tensor(os.path.join(root, hparams["path_extract"], hparams["raw_dataset"]))
    log.info("Image loading and transformation complete.")
    
    log.info("Saving dataset...")
    torch.save(training_images, os.path.join(root, hparams["processed_dataset"], 'fruit_training_images.pt'))
    torch.save(training_labels, os.path.join(root, hparams["processed_dataset"], 'fruit_training_labels.pt'))
    log.info("Dataset saved.")

def retrieve_from_api(path_extract, kaggle_dataset):

    log.info(f"Authenticating API for dataset: {kaggle_dataset}")
    api = KaggleApi()
    api.authenticate()
    log.info("API authentication successful.")
    
    log.info(f"Downloading files for dataset: {kaggle_dataset}")
    api.dataset_download_files(kaggle_dataset , os.path.join(path_extract) , 'zips')
    log.info("File download complete.")
    
    log.info("Extracting files...")
    with zipfile.ZipFile(os.path.join(path_extract, 'zips/fruits.zip'), 'r') as zip_ref:
        zip_ref.extractall(path_extract)
    log.info("File extraction complete.")
    
def images_to_tensor(directory, transform=None):
    log.info(f"Converting images in {directory} to tensors...")
    tensor_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.jpg'):
                path = os.path.join(root, filename)
                image = Image.open(path)
                if transform is not None:
                    image = transform(image)
                tensor_list.append(image)
    log.info(f"Converted {len(tensor_list)} images to tensors.")
    return torch.stack(tensor_list)

def show_image(tensor, index):
    # Select the image by index
    img_tensor = tensor[index]
    
    # If the image has 3 channels, rearrange dimensions so it's suitable for displaying
    if img_tensor.shape[0] == 3:
        img_tensor = img_tensor.permute(1, 2, 0)
    
    plt.imshow(img_tensor)
    plt.show()
   
def labels_to_tensor(directory, dataset_name="."): 
    log.info(f"Converting labels in {directory} to tensors...")
    labels = []
    folder_names = []
    i = 0
    
    if dataset_name == ".":
        folder_names = os.listdir(directory) 
        # Loop over all subdirectories in the root directory
        for subdir in folder_names:
            subdir_path = os.path.join(directory, subdir)
            
            # Check if it's a directory  
            if os.path.isdir(subdir_path):
                # Count the number of files in the subdirectory
                num_files = len(os.listdir(subdir_path))
                # Fill 
                index_n = [i] * num_files
                i += 1
                labels += index_n
        log.info(f"Converted labels for {len(folder_names)} subdirectories to tensors.")
    else:
        subdir_path = os.path.join(directory, dataset_name)
        num_files = len(os.listdir(subdir_path))
        index_n = [i] * num_files
        i += 1
        labels += index_n
        log.info(f"Converted labels for 1 subdirectory to tensors.")
        
    # Concatenate all tensors in the list
    log.info(f"Converted {len(labels)} labels to tensors.")
    return torch.Tensor(labels)

if __name__ == '__main__':
    main()
