
import re
import cv2
import random
import datetime
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import torch.utils.data as data

from .utils import is_image_file, check_exceptions

class USNeedleDataset(data.Dataset):
    def __init__(self, root_dir, transform=None, extensions=None, resize_img=None):
        super(USNeedleDataset, self).__init__()
        self.resize_img = resize_img
        self.transform = transform

        image_dirs = listdir(join(root_dir,'images*'))
        shaft_dirs = listdir(join(root_dir, 'shafts*'))
        tip_dirs   = listdir(join(root_dir, 'tips*'))

        self.total_image_filenames = []
        self.total_shaft_filenames = []
        self.total_tip_filenames = []
        for image_dir in image_dirs:
            srt_splt = re.split('/|\.', image_dir)
            shaft_dir = listdir(join(root_dir, 'shafts'+ srt_splt[-2]))
            tip_dir = listdir(join(root_dir, 'tips'  + srt_splt[-2]))
            for image_file in listdir(image_dir):
                shaft_files = listdir(shaft_dir)
                tip_files = listdir(tip_dir)
                srt_splt = re.split('image', image_file)
                shaft_file = "shaft"+srt_splt[-2]+srt_splt[-1]
                tip_file = "tip"+srt_splt[-2]+srt_splt[-1]
                shft_ix = shaft_files.index(shaft_file)
                tip_ix = tip_files.index(tip_file)

                if shft_ix is not None and tip_ix is not None:
                    self.total_image_filenames.append(image_file)
                    self.total_shaft_filenames.append(shaft_file)
                    self.total_image_filenames.append(tip_file)
        assert len(self.total_image_filenames) == len(self.total_shaft_filenames)
        assert len(self.total_image_filenames) == len(self.total_tip_filenames)

        print("Number of train images: {}".format(self.__len__()+4))


        def __getitem__(self, index):
            np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

            hsv = np.zeros(np.concatenate((3,), tuple(self.resize_img)), dtype=np.float32)
            # target = np.zeros(np.concatenate(tuple(self.resize_img), (2,)), dtype=np.float32)

            diff_index = random.randint(1,3)
            if random.random()>0.5:
                diff_index = -diff_index

            cur_index = index + 2
            prev_index = cur_index + diff_index

            curr_frame = cv2.imread(self.total_image_filenames[cur_index])
            prev_frame = cv2.imread(self.total_image_filenames[prev_index])

            shaft_seg = cv2.imread(self.total_shaft_filename[cur_index])
            tip_seg = cv2.imread(self.total_tip_filename[cur_index])

            if resize_img is not None:
                curr_frame_resized = cv2.resize(curr_frame, dsize=resize_img, interpolation=cv2.INTER_AREA)
                prev_frame_resized = cv2.resize(prev_frame, dsize=resize_img, interpolation=cv2.INTER_AREA)
                shaft_seg_resized  = (shaft_seg - np.min(shaft_seg)) / (np.max(shaft_seg) - np.min(shaft_seg))
                shaft_seg_resized  = cv2.resize(shaft_seg, dsize=resize_img, interpolation=cv2.INTER_AREA)
                shaft_seg_resized  = np.array(shaft_seg_resized > 0.5, dtype=int)
                tip_seg_resized    = (tip_seg - np.min(tip_seg)) / (np.max(tip_seg) - np.min(tip_seg))
                tip_seg_resized    = cv2.resize(tip_seg,    dsize=resize_img, interpolation=cv2.INTER_AREA)
                tip_seg_resized    = np.array(tip_seg_resized > 0.5, dtype=int)

            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[0, ...] = ang * 180 / np.pi / 2
            hsv[1, ...] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[2, ...] = curr_frame
            hsv1 = np.uint8(hsv)
            bgr = cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)

            input  = Image.fromarray(bgr)
            target = np.concatenate((shaft_seg_resized,tip_seg_resized), axis=0)

            # handle exceptions
            check_exceptions(input, target)
            if self.transform:
                input, target = self.transform(input, target)

            return input, target

    def __len__(self):
        return len(self.total_image_filenames-4)