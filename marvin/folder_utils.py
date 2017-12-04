from os import mkdir
from os.path import exists, split

def ensure_folder_exists(folder):
    if not exists(folder):
        ensure_folder_exists(split(folder)[0])
        try:
            mkdir(folder)
        except OSError as e:
            if e.errno != 17:
                raise e