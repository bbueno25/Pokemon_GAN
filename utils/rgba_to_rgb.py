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
    info('images:converting to rgb')
    src_path = path.join(DIR_NAME, 'resized_images')
    dst_path = path.join(DIR_NAME, 'rgb_images')
    if not path.exists(dst_path):
        mkdir(dst_path)
    for filename in listdir(src_path):
        img = Image.open(path.join(src_path, filename))
        if img.mode == 'RGBA':
            info('images:converting to rgb:{}'.format(filename))
            img.load()    # required for png.split()
            split_img = Image.new('RGB', img.size, (0, 0, 0))
            split_img.paste(img, mask=img.split()[3])    # 3 is the alpha channel
            split_img.save(path.join(dst_path, filename.split('.')[0] + '.jpg'), 'JPEG')
        else:
            info('images:converting to jpg:{}'.format(filename))
            img = img.convert('RGB')
            img.save(path.join(dst_path, filename.split('.')[0] + '.jpg'), 'JPEG')
    info('images:conversion completed')
