from tests import _PATH_DATA
import os
import torch
from src.data.make_dataset import images_to_tensor, labels_to_tensor
import torchvision.transforms as transforms
import pytest
#from kaggle.api.kaggle_api_extended import KaggleApi
import unittest
from unittest.mock import patch, MagicMock

# Path: tests/test_data.py
print(_PATH_DATA)
path_to_data = _PATH_DATA+"/raw/fruits-360_dataset/fruits-360/Training"


@pytest.mark.skipif(not os.path.exists(path_to_data), reason="Data files not found")
def test_images_to_tensor():
    """ Test if images are converted to tensors correctly """
    # Arrange
    path_extract = os.path.join(_PATH_DATA, "raw")

     # Provide the paths of the existing images
    image_paths = path_extract + "/fruits-360_dataset/fruits-360/Training/Apricot"
    
    # Call the function to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    tensor_list = images_to_tensor(image_paths, transform=transform)

    # Check if the length of the tensor list matches the number of image files
    assert len(tensor_list) == 246 #TODO for all 67692

    # Check if the first tensor in the list has the correct shape
    assert isinstance(tensor_list[0], torch.Tensor)
    assert tensor_list[0].shape == torch.Size([3, 100, 100])


# class TestRetrieveFromApi(unittest.TestCase):

#     @patch('src.data.make_dataset.KaggleApi',autospec= True) 
#     @patch('src.data.make_dataset.zipfile.ZipFile',autospec= True)  
#     def test_retrieve_from_api(self, mock_kaggle_api, mock_zipfile):
#             """ Test if files are downloaded and extracted using mock objects """
#             # Mocking objects
#             mock_api_instance = MagicMock()
#             mock_kaggle_api.return_value = mock_api_instance
#             mock_zip_instance = MagicMock()
#             mock_zipfile.return_value = mock_zip_instance

#             # Mock the API methods and attributes
#             mock_api_instance.authenticate.return_value = None
#             mock_api_instance.dataset_download_files.return_value = None

#             # Call the function
#             path_extract = os.path.join(_PATH_DATA, "raw")
#             kaggle_dataset = 'moltean/fruits'
#             retrieve_from_api(path_extract, kaggle_dataset)

#             # Assertions
#             mock_kaggle_api.assert_called_once()
        

if __name__ == '__main__':
    unittest.main()