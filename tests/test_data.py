from tests import _PATH_DATA
import os
from src.data.make_dataset import retrieve_from_api


# Path: tests/test_data.py
print(_PATH_DATA)
def test_retrieve_from_api():
    # Arrange
    path_extract = os.path.join(_PATH_DATA, "raw")
    kaggle_dataset = "moltean/fruits"
    
    # Act
    #retrieve_from_api(path_extract, kaggle_dataset)
    
    # Assert
    assert os.path.exists(os.path.join(path_extract, "zips/fruits.zip"))
    #assert os.path.exists(os.path.join(path_extract, "fruits-360"))