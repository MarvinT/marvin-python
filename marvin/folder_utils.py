from os import mkdir
from os.path import exists, split

def ensure_folder_exists(folder):
    if not exists(folder):
        ensure_folder_exists(split(folder)[0])
        mkdir(folder)