import os

data_dir = "Y:/data/"
if not os.path.exists(data_dir):
    data_dir = "/cifs/diedrichsen/data/"
if not os.path.exists(data_dir):
    data_dir = "/Volumes/diedrichsen_data$/data"

save_dir = 'C:/Users/barafat/Dropbox (Personal)/Papers/MultiTaskBattery_paper/Figures'
if not os.path.exists(save_dir):
    save_dir = '/Users/jdiedrichsen/Dropbox/papers/MultiTaskBattery_paper/Figures'


repo_dir = os.path.dirname(os.path.abspath(__file__))
