import os
import torch
import torchvision.transforms as transforms
import zipfile
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

def retrieve_from_api(path_extract):
    path_zip =f"{path_extract}/zips"
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("moltean/fruits" , path_zip)
    
    with zipfile.ZipFile(f"{path_zip}/fruits.zip", 'r') as zip_ref:
    # Extract all contents to the specified directory
        zip_ref.extractall(path_extract)
    
def images_to_tensor(directory):
    tensor_list = []
    for root, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more conditions if there are other image types
                img_path = os.path.join(root, filename)
                img = Image.open(img_path)
                tensor = transform(img)
                tensor_list.append(tensor)
    return torch.stack(tensor_list)  # Stacks the list of tensors along a new dimension

def show_image(tensor, index):
    # Select the image by index
    img_tensor = tensor[index]
    
    # If the image has 3 channels, rearrange dimensions so it's suitable for displaying
    if img_tensor.shape[0] == 3:
        img_tensor = img_tensor.permute(1, 2, 0)
    
    plt.imshow(img_tensor)
    plt.show()
   
def labels_to_tensor(directory): 
    labels = []
    folder_names = []
    i = 0
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
    
    # Concatenate all tensors in the list
    return torch.Tensor(labels)

if __name__ == '__main__':
    # Get the data and process it
    path_extract = "./data/raw"
    retrieve_from_api(path_extract)

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,))])
    
    # Load the images from the folder and apply the transformation
    training_images = images_to_tensor('./data/raw/fruits-360_dataset/fruits-360/Training/')
    training_labels = labels_to_tensor('./data/raw/fruits-360_dataset/fruits-360/Training/')
    # Save dataset
    torch.save(training_images, "./data/processed/fruit_training_images.pt")
    torch.save(training_labels, "./data/processed/fruit_training_labels.pt")
    
    # Show one of the images
    print(training_images.shape)
    print(training_labels.shape)
    
    pass
