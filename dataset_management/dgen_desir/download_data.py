import pathlib
import urllib.request
import zipfile

"""
Downloads the DGEN 380 turbofan test bench from https://zenodo.org/records/13860307 
Note: The there is inconsistency in the capitalisation of the file names in the dataset. The file names are converted to lowercase
"""

# save_path = pathlib.Path("/home/douwm/data/dgen_desir")
save_path = pathlib.Path("/home/douwm/data/DGEN380_turbofan")
# make new folder for raw data if it
if not save_path.exists():
    save_path.mkdir()

download_link = "https://zenodo.org/api/records/13860307/files-archive"
# Download the data for each url
print("Downloading: ", download_link)
for retries in range(5):  # In case something goes wrong, try again
    try:
        urllib.request.urlretrieve(download_link, save_path.joinpath("dgen_desir.zip"))
        break
    except:
        if retries == 5:
            raise Exception("Could not download file: ", download_link)
        print("Retrying: ", download_link)
        pass

# Unzip the file
with zipfile.ZipFile(save_path.joinpath("dgen_desir.zip"), 'r') as zip_ref:
    zip_ref.extractall(save_path)

# Clean up the zip file
save_path.joinpath("dgen_desir.zip").unlink()
print("Data downloaded and saved to: ", save_path)

# Rename all files to lowercase
for file in save_path.glob("**/*"):
    if file.is_file():
        file.rename(file.parent.joinpath(file.name.lower()))

print("Warning: Original file names are inconsistent in capitalisation. All files are converted to lowercase")
