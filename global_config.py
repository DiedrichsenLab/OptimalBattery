import os

data_path = "Y:/data/"
if not os.path.exists(data_path):
    data_path = "/cifs/diedrichsen/data/"
if not os.path.exists(data_path):
    data_path = "/Volumes/diedrichsen_data$/data"
