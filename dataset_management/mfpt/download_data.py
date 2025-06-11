import os.path
import zipfile
import pathlib
import requests
from tqdm import tqdm

"""Downloads MFPT Fault Data Sets from the internet and saves it to the raw_data directory"""

# Save in parent directory of this file
save_path = pathlib.Path(__file__).parent.joinpath("raw_data")

def download_and_extract_mfpt_data():
    """
    Downloads and extracts the MFPT Fault Data Sets from the MFPT website.
    The data is saved to the raw_data directory in the same folder as this script.
    """
    # Create the download path directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # URL for the MFPT dataset
    url = "https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"
    zip_file_path = os.path.join(save_path, "mfpt_data.zip")
    
    # Download the file with proper headers to mimic a browser request
    print(f"Downloading MFPT dataset from {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.mfpt.org/',
        'Connection': 'keep-alive',
    }
    
    for retries in range(5):  # In case something goes wrong, try again
        try:
            print(f"Downloading to {zip_file_path}")
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            # Download with progress bar
            with open(zip_file_path, 'wb') as file, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)
            break
        except Exception as e:
            if retries == 4:
                raise Exception(f"Could not download file: {url}. Error: {e}")
            print(f"Retrying: {url}")
            pass
    
    # Extract the zip file
    print("Extracting files...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
    
    # Remove the zip file to save space (optional)
    print("Removing zip file to save space...")
    os.remove(zip_file_path)
    
    print(f"Data downloaded and extracted to: {save_path}")

if __name__ == "__main__":
    download_and_extract_mfpt_data()