# from tests import _PATH_DATA
import os
from make_dataset import retrieve_from_api

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

#from make_dataset import retrieve_from_api

# Path: tests/test_data.py
print(_PATH_DATA)
def test_retrieve_from_api():
    # Arrange
    path_extract = os.path.join(_PATH_DATA, "extract")
    kaggle_dataset = "moltean/fruits"
    
    # Act
    retrieve_from_api(path_extract, kaggle_dataset)
    
    # Assert
    assert os.path.exists(os.path.join(path_extract, "zips/fruits.zip"))
    assert os.path.exists(os.path.join(path_extract, "fruits-360"))