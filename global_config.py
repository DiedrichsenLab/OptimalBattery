import os

data_path = "Y:/data/"
if not os.path.exists(data_path):
    data_path = "/cifs/diedrichsen/data/"
if not os.path.exists(data_path):
    data_path = "/Volumes/diedrichsen_data$/data"

save_dir = 'C:/Users/barafat/Dropbox (Personal)/Papers/MultiTaskBattery_paper/Figures'
if not os.path.exists(save_dir):
    save_dir = '/Users/jdiedrichsen/Dropbox/papers/MultiTaskBattery_paper/Figures'