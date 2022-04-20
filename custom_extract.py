import os

import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('data/custom_dataset/')
    zip_ref.close()


if __name__ == "__main__":
    if not os.path.isdir('data/custom_dataset/train.zip'):
        extract('/home/ahmadob/dataset/facerecognition_dataset/train.zip')