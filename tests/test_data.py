from tests import _PATH_DATA
import os
import torch
import responses
from src.data.make_dataset import retrieve_from_api, images_to_tensor, labels_to_tensor
import torchvision.transforms as transforms
import pytest


# Path: tests/test_data.py
print(_PATH_DATA)
path_to_data = _PATH_DATA+"/raw/fruits-360_dataset/fruits-360/Training"


@pytest.mark.skipif(not os.path.exists(path_to_data), reason="Data files not found")
def test_images_to_tensor():
    """ Test if images are converted to tensors correctly """
    # Arrange
    path_extract = os.path.join(_PATH_DATA, "raw")

     # Provide the paths of the existing images
    image_paths = path_extract + "/fruits-360_dataset/fruits-360/Training/Apple Braeburn"
    
    # Call the function to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    tensor_list = images_to_tensor(image_paths, transform=transform)

    # Check if the length of the tensor list matches the number of image files
    assert len(tensor_list) == 492 #TODO for all 67692

    # Check if the first tensor in the list has the correct shape
    assert isinstance(tensor_list[0], torch.Tensor)
    assert tensor_list[0].shape == torch.Size([3, 100, 100])


# def test_retrieve_from_api():
#     # Arrange
#     path_extract = os.path.join(_PATH_DATA, "raw")
#     kaggle_dataset = "moltean/fruits"

#     with responses.RequestsMock() as rsps:
#         rsps.add(
#             responses.GET,
#             "https://www.kaggle.com/api/v1/datasets/list",
#             json={"data": "your-mock-data"},
#             status=200,
#         )

#         # Call the function to retrieve data
#         path_extract = str(tmpdir.mkdir("path_extract"))
#         retrieve_from_api(path_extract, kaggle_dataset)

#         # Check if files are downloaded and extracted
#         assert os.path.exists(os.path.join(path_extract, "fruits.zip"))
#         assert os.path.exists(os.path.join(path_extract, "fruits"))
    
    # Assert
    # assert os.path.exists(os.path.join(path_extract, "zips/fruits.zip"))