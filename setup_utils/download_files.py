import requests
import tarfile
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

# Use the flags below to indicate what you want to download.
get_data = True # ~28 MB
get_data_SI = False # ~300 MB
get_model_actv = False # ~1.6 GB
get_regr_weights = False # ~80 KB

print(f'Downloading files to {ROOT}')

def download_extract_remove(url, extract_location):
    """
    Base function from Jenelle Feather: https://github.com/jenellefeather/model_metamers_pytorch/blob/master/download_large_files.py
    """
    temp_file_location = os.path.join(extract_location, 'temp.tar')
    print('Downloading %s to %s'%(url, temp_file_location))
    with open(temp_file_location, 'wb') as f:
        r = requests.get(url, stream=True)
        for chunk in r.raw.stream(1024, decode_content=False):
            if chunk:
                f.write(chunk)
                f.flush()
    print('Extracting %s'%temp_file_location)
    tar = tarfile.open(temp_file_location)
    # Check if there is the extraction would overwrite an existing file
    for member in tar.getmembers():
        if os.path.exists(member.name):
            print('File %s already exists, aborting'%member.name)
            sys.exit()

    tar.extractall(path=extract_location) # untar file into same directory
    tar.close()

    print('Removing temp file %s'%temp_file_location)
    os.remove(temp_file_location)

# Download the data folder
if get_data:
    url_data_folder = 'https://evlabwebapps.mit.edu/public_data/tuckute2023_driving_suppressing/data.tar'
    download_extract_remove(url_data_folder, ROOT)

# Download the data SI folder
if get_data_SI:
    url_data_SI_folder = 'https://evlabwebapps.mit.edu/public_data/tuckute2023_driving_suppressing/data_SI.tar'
    download_extract_remove(url_data_SI_folder, ROOT)

# Download the GPT2-XL model activations. To the baseline set: beta-control-neural-T
# For drive/suppress along with baseline for the search approach: beta-control-neural-D
# OBS! Large file (~1.6 GB)
if get_model_actv:
    print(f'OBS! Large file (~1.6 GB)')
    url_model_actv = 'https://evlabwebapps.mit.edu/public_data/tuckute2023_driving_suppressing/model-actv.tar'
    download_extract_remove(url_model_actv, ROOT)

# Download the regression weights fitted on the baseline set
if get_regr_weights:
    url_regr_weights = 'https://evlabwebapps.mit.edu/public_data/tuckute2023_driving_suppressing/regr-weights.tar'
    download_extract_remove(url_regr_weights, ROOT)


