import os
import pathlib
import requests
from tqdm import tqdm
import subprocess


def download_dataset():
    """Download the University of Paderborn bearing dataset."""
    data_path = pathlib.Path(__file__).parent.joinpath("raw_data")
    os.makedirs(data_path, exist_ok=True)

    base_url = "https://groups.uni-paderborn.de/kat/BearingDataCenter"

    # Complete dataset including all bearing conditions
    # K series: healthy bearings
    # KA series: artificial damage type A
    # KB series: artificial damage type B
    # KI series: artificial damage type I
    files = [
        "K001.rar",
        "K002.rar",
        "K003.rar",
        "K004.rar",
        "K005.rar",
        "K006.rar",
        "KA01.rar",
        "KA03.rar",
        "KA04.rar",
        "KA05.rar",
        "KA06.rar",
        "KA07.rar",
        "KA08.rar",
        "KA09.rar",
        "KA15.rar",
        "KA16.rar",
        "KA22.rar",
        "KA30.rar",
        "KB23.rar",
        "KB24.rar",
        "KB25.rar",
        "KB26.rar",
        "KB27.rar",
        "KI01.rar",
        "KI03.rar",
        "KI04.rar",
        "KI05.rar",
        "KI07.rar",
        "KI08.rar",
        "KI14.rar",
        "KI16.rar",
        "KI17.rar",
        "KI18.rar",
        "KI21.rar",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }

    for filename in files:
        file_path = data_path.joinpath(filename)

        # Skip if file already exists
        if file_path.exists():
            print(f"Skipping {filename} - already exists")
            continue

        url = f"{base_url}/{filename}"
        print(f"Downloading {filename}...")

        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(file_path, "wb") as f,
                tqdm(
                    desc=filename,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            continue

    print("\nExtracting RAR files...")
    for filename in files:
        rar_path = data_path.joinpath(filename)
        if rar_path.exists():
            try:
                # Extract to a subdirectory named after the file
                extract_dir = data_path.joinpath(filename.replace(".rar", ""))
                os.makedirs(extract_dir, exist_ok=True)

                # Use system unrar command
                result = subprocess.run(
                    ["unrar", "x", str(rar_path), str(extract_dir) + "/"],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    print(f"Extracted {filename}")
                else:
                    print(f"Error extracting {filename}: {result.stderr}")

            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                print("Note: Make sure 'unrar' is installed on your system")


if __name__ == "__main__":
    download_dataset()
