import os
from os.path import join


def get_dataset_files(path, exclude=None, only=None):
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-3:] != 'hd5':
                continue
            exclude_file = False
            if only:
                exclude_file = True
                for pattern in only:
                    if file == str(pattern) + '.hd5':
                        exclude_file = False
                        break
            elif exclude:
                exclude_file = False
                for ex in exclude:
                    if file == str(ex) + '.hd5':
                        exclude_file = True
                        break
            if exclude_file:
                continue
            filelist.append(join(root, file))

    return filelist