import gdown
import zipfile
import os

def download_and_unzip(url, output, extract_to):
    # Download the file
    print(f'Downloading {output} from {url}')
    gdown.download(url, output, quiet=False)

    # Unzip the file
    print(f'Unzipping {output} to {extract_to}')
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Clean up zip file
    os.remove(output)
    print(f'Download and extraction complete. Data available in {extract_to}')

def main():
    # URL and output details
    url = 'https://drive.google.com/uc?id=1HAk0yiiCKUT47i6l6kTW7Z3yovYCBdJW'
    output = 'vfm_assignment.zip'
    extract_to = 'vfm_assignment'
    
    # Create directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Download and unzip the file
    download_and_unzip(url, output, extract_to)
    
    # Check contents of the directory
    contents = os.listdir(extract_to)
    print(f'Contents of {extract_to}: {contents}')

if __name__ == "__main__":
    main()
