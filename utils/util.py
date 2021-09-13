
import os
import json
import csv
import numpy as np
from os import mkdir, makedirs
import collections
from skimage.exposure import rescale_intensity

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imgtype='img', datatype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.ndim == 4:  # image_numpy (C x W x H x S)
        mid_slice = image_numpy.shape[-1]//2
        image_numpy = image_numpy[:,:,:,mid_slice]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3,1,1))
    image_numpy = np.transpose(image_numpy, (1,2,0))

    if imgtype == 'img':
        image_numpy = (image_numpy + 8) / 16.0 * 255.0
    if np.unique(image_numpy).size == int(1):
        return image_numpy.astype(datatype)
    return rescale_intensity(image_numpy.astype(datatype))

def json_file_to_pyobj(filename):
    def _json_object_hook(d):
        return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data):
        return json.loads(data,object_hook=_json_object_hook)
    return json2obj(open(filename).read())

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        makedirs(paths, exist_ok=True)


def csv_write(out_filename, in_header_list, in_val_list):
    with open(out_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(in_header_list)
        writer.writerows(zip(*in_val_list))