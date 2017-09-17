from os import mkdir
from os.path import exists

def ensure_folder_exists(folder):
    if not exists(folder):
        mkdir(folder)