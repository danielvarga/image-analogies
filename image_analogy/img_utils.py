import numpy as np
from scipy.misc import imread, imresize

from . import vgg16


# util function to open, resize and format pictures into appropriate tensors
def load_image(image_path, remove_alpha=True):
    im = imread(image_path)
    if im.shape[2]==4 and remove_alpha:
        im = im[:, :, :3]
        print('WARNING: Removed alpha channel from image {}'.format(image_path))
    return im

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(x, img_width, img_height, do_conditioning=True):
    img = imresize(x, (img_height, img_width), interp='bicubic').astype('float64')
    if do_conditioning:
        img = vgg16.img_to_vgg(img)
    else:
        img = img[:,:,::-1]
        img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x, contrast_percent=0.0, resize=None):
    x = vgg16.img_from_vgg(x)
    if contrast_percent:
        min_x, max_x = np.percentile(x, (contrast_percent, 100 - contrast_percent))
        x = (x - min_x) * 255.0 / (max_x - min_x)
    x = np.clip(x, 0, 255)
    if resize:
        x = imresize(x, resize, interp='bicubic')
    return x.astype('uint8')
