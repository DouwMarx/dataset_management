import os.path
import urllib.request
import re
from tqdm import tqdm
from urllib3 import Retry, PoolManager


"""This script downloads the Case western reserve 12000Hz data from the internet and saves it to the data directory"""

# For the drive-end measurements, retrieve the web page in html format
drive_end_url = "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data"
download_path = "/home/douwm/data/CWR/drive_end_12k"

# Create the download path directory if it does not exist
if not os.path.exists(download_path):
    os.makedirs(download_path)

with urllib.request.urlopen(drive_end_url) as f:
    html = f.read().decode('utf-8')

# Find all the strings that end with .mat in the web page
mat_files = re.findall(r'href=".*\.mat"', html)

# Remove the href=" and " from the strings
mat_files = [re.sub(r'href="|"', '', mat_file) for mat_file in mat_files]

print(mat_files)

# Download the files and save them to the download_path with the same name as in the web page
for file_url in tqdm(mat_files):
    print("Downloading: ", file_url)
    for retries in range(5): # In case something goes wrong, try again
        try:
         urllib.request.urlretrieve(file_url, os.path.join(download_path,os.path.basename(file_url)))
         break
        except:
           print("Retrying: ", file_url)
           pass

    # retries = Retry(connect=5, read=2, redirect=5, backoff_factor=0.1)
    # http = PoolManager(retries=retries)
    # response = http.request(file_url, os.path.join(download_path,os.path.basename(file_url)))
