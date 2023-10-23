import os.path
import zipfile
import py7zr
from pyunpack import Archive

import urllib.request

from tqdm import tqdm

"""This script downloads and extracts the nasa IMS data from the internet and saves it to the data directory"""


def get_all_data_from_nasa_website():
    download_path = "/home/douwm/data/ims"

    # Create the download path directory if it does not exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    url = "https://data.nasa.gov/download/7wwx-fk77/application%2Fzip"
    url = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip" # Mirror
    # Download the files and save them to the download_path with the same name as in the web page
    print("Downloading: ", url)
    for retries in range(5):  # In case something goes wrong, try again
        try:
            urllib.request.urlretrieve(url,  os.path.join(download_path, "ims.zip"))
            break
        except:
            if retries == 4:
                raise Exception("Could not download file: ", url)
            print("Retrying: ", url)
            pass

    print("Extracting files")
    # Unzip the file
    with zipfile.ZipFile(os.path.join(download_path, "ims.zip"), 'r') as zip_ref:
        zip_ref.extractall(download_path)

    # # Unzip the 7z file IMS.7z
    with py7zr.SevenZipFile(os.path.join(download_path, "4. Bearings", "IMS.7z"), mode='r') as z:
        z.extractall(path=download_path)
    # Delete the ims.zip file to save space
    os.remove(os.path.join(download_path, "4. Bearings", "IMS.7z"))
    # Delete the 4. Bearings folder
    os.rmdir(os.path.join(download_path, "4. Bearings"))

    # # Extact all .rar files in the download path
    for file in tqdm(os.listdir(download_path)):
        if file.endswith(".rar"):
            Archive(os.path.join(download_path, file)).extractall(download_path)

    # The folder structure for the 3rd test is different: 4th_test/txt/... Rename to 3rd_test/... to be consistent with readme
    os.rename(os.path.join(download_path, "4th_test", "txt"), os.path.join(download_path, "3rd_test"))
    # Delete the 4th_test folder
    os.rmdir(os.path.join(download_path, "4th_test"))

    # Remove all the .rar files to save space
    for file in os.listdir(download_path):
        if file.endswith(".rar"):
            os.remove(os.path.join(download_path, file))


if __name__ == "__main__":
    get_all_data_from_nasa_website()
