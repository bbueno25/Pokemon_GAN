from logging import basicConfig
from logging import INFO
from logging import info
from os import getcwd
from os import listdir
from os import mkdir
from os import path
from PIL import Image

basicConfig(level=INFO)

if __name__ == '__main__':
    DIR_NAME = path.join(getcwd(), 'data')
    info('images:resizing')
    src_path = path.join(DIR_NAME, 'original_images')
    dst_path = path.join(DIR_NAME, 'resized_images')
    if not path.exists(dst_path):
        mkdir(dst_path)
    for filename in listdir(src_path):
        img = Image.open(path.join(src_path, filename))
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(path.join(dst_path, filename))
        info('images:resizing:{}'.format(filename))
    info('images:resizing completed')
