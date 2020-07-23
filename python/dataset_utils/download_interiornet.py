"""
A utility script to download the InteriorNet dataset.
"""
import os
import csv
import gdown
from zipfile import ZipFile

# The list with all the files and links in the HD7 folder of the InteriorNet dataset.
list_csv = 'list_of_files_inHD7.csv'

dataset = '/nvme/datasets/interiornet'

files_downloaded = [name for name in os.listdir(dataset)
                    if os.path.isdir(os.path.join(dataset, name))]

max_num = 6000

print("Starting download.")
with open(list_csv, 'r') as csvfile:
    scene_reader = csv.reader(csvfile, delimiter=',')
    for scene in scene_reader:
        name = scene[0][:-4] # cut out .zip
        url = scene[1]
        if name in files_downloaded:
            print("{} already downloaded. Skipping.".format(name))
            continue
        print("Downloading " + name + '.zip')
        output = os.path.join(dataset, name + '.zip')
        gdown.download(url, output, quiet=False)
        if os.path.isfile(output):
            with ZipFile(output, 'r') as zipObj:
                # Extract all the contents of zip file in different directory
                zipObj.extractall(dataset)
            os.remove(output)
            files_downloaded = [name for name in os.listdir(dataset)
                                if os.path.isdir(os.path.join(dataset, name))]
        if len(files_downloaded) > max_num:
            break
