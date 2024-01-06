import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == '__main__':
    # Get the data and process it
    path_extract = "./data/raw"
    path_zip =f"{path_extract}/zips"
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("moltean/fruits" , path_zip)
    
    with zipfile.ZipFile(f"{path_zip}/fruits.zip", 'r') as zip_ref:
    # Extract all contents to the specified directory
        zip_ref.extractall(path_extract)
    pass