import os.path
import urllib.request
import re
from tqdm import tqdm
import pathlib

"""Downloads Case western reserve 12kHz and 48KHz data from the internet and saves it to the data prescribed below"""
# Save in parent directory of this file

# save_path = "/home/douwm/data/CWR"

# make new folder for raw data if it does not exist
save_path = pathlib.Path(__file__).parent.joinpath("raw_data")
if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_all_mat_data_from_cwr_page(download_path, page_url):
    # Create the download path directory if it does not exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    with urllib.request.urlopen(page_url) as f:
        html = f.read().decode('utf-8')

    # Find all the strings that end with .mat in the web page
    mat_files = re.findall(r'href=".*\.mat"', html)

    # Remove the href=" and " from the strings
    mat_files = [re.sub(r'href="|"', '', mat_file) for mat_file in mat_files]

    print("Found the following files .mat files: ")
    print(mat_files)

    # Download the files and save them to the download_path with the same name as in the web page
    for file_url in tqdm(mat_files):
        print("Downloading: ", file_url)
        for retries in range(5):  # In case something goes wrong, try again
            try:
                urllib.request.urlretrieve(file_url, os.path.join(download_path, os.path.basename(file_url)))
                break
            except:
                if retries == 50:
                    raise Exception("Could not download file: ", file_url)
                print("Retrying: ", file_url)
                pass

# Define the urls to download the data from
data_12k_de_bearing_url = "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data"
data_48k_de_bearing_url = "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data"
data_12k_fe_bearing_url = "https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data"
data_48k_normal = "https://engineering.case.edu/bearingdatacenter/normal-baseline-data"

# Download the data for each url
for download_url in [data_12k_de_bearing_url, data_48k_de_bearing_url, data_12k_fe_bearing_url, data_48k_normal]:
    get_all_mat_data_from_cwr_page(save_path, download_url)

print("Data downloaded and saved to: ", save_path)