
import torchsample.transforms as ts
from pprint import pprint
import dataio.transformation.pytorch_transforms as pt

class Transformations:

    def __init__(self, name):
        self.name = name

        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.random_flip_prob = 0.0

    def get_transformation(self):
        return {
            'us_needle': self.us_needle_transform,
            'od_maskrcnn': self.od_maskrcnn_transform,
        }[self.name]()

    def print(self):
        print('\n\n########### augmentation parameters ###########')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts):
        t_opts = getattr(opts, self.name)

        if hasattr(t_opts, 'shift'):
            self.shift_val=t_opts.shift
        if hasattr(t_opts, 'rotate'):
            self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'random_flip_prob'):
            self.random_flip_prob = t_opts.random_flip_prob

    def us_needle_transform(self):
        train_transform = ts.Compose(
            [ts.ToTensor(),
             ts.ChannelsFirst(),
             ts.TypeCast(['float', 'float']),
             ts.RandomFlip(h=False, v=True, p=self.random_flip_prob),
             ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
                             interp=('bilinear')),
             ts.NormalizeMedicPercentile(norm_flag=(True, False))
             ])

        valid_transform = ts.Compose(
            [ts.ToTensor(),
             ts.ChannelsFirst(),
             ts.TypeCast(['float', 'float']),
             ts.NormalizeMedicPercentile(norm_flag=(True,False))
             ])

        return {'train': train_transform, 'valid': valid_transform}

    def od_maskrcnn_transform(self):
        train_transform = pt.Compose(
            [pt.ToTensor(),
             pt.RandomHorizontalFlip(0.5)
             ])

        valid_transform = pt.Compose(
            [pt.ToTensor()])

        return {'train': train_transform, 'valid': valid_transform}