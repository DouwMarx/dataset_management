import os.path
import urllib.request
import re
from tqdm import tqdm
from urllib3 import Retry, PoolManager

"""This script downloads the cao_zhang_tang data from figshare and saves it to the data directory"""


def get_all_mat_data_from_figshare():
    download_path = "/home/douwm/data/cao_zhang_tang"

    # Create the download path directory if it does not exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    urls = {"time_domain": "https://figshare.com/ndownloader/files/11053469",
            "order_domain": "https://figshare.com/ndownloader/files/11053466"}

    # Download the files and save them to the download_path with the same name as in the web page
    for data_name, file_url in tqdm(urls.items()):
        print("Downloading: ", file_url)
        for retries in range(5):  # In case something goes wrong, try again
            try:
                export_path = os.path.join(download_path, data_name) + ".mat"
                urllib.request.urlretrieve(file_url, export_path)
                break
            except:
                if retries == 4:
                    raise Exception("Could not download file: ", file_url)
                print("Retrying: ", file_url)
                pass


if __name__ == "__main__":
    get_all_mat_data_from_figshare()
