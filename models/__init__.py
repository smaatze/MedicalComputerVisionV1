
import os
from .networks import get_network

class ModelOpts:
    def __init__(self):
        self.gpu_ids = [0]
        self.is_train = True
        self.continue_train = False
        self.which_epoch = int(0)
        self.save_dir = '.\\checkpoints\\default'
        self.model_type = 'seg'
        self.model_name = 'unet'
        self.input_nc = 1
        self.output_nc = 2
        self.lr_rate = 1e-12
        self.l2_reg_weight = 0.0
        self.feature_scale = 4
        self.tensor_dim = '2d'
        self.path_pretrain_model = None
        self.criterion = 'cross_entropy'


    def initialise(self, json_model_opts):
        opts = json_model_opts

        self.raw = json_model_opts
        self.model_name = opts.model_name
        self.gpu_ids = opts.gpu_ids
        self.is_train = opts.is_train
        self.save_dir = os.path.join(opts.checkpoints_dir, opts.experiment_name)
        self.model_type = opts.model_type
        self.input_nc = opts.input_nc
        self.output_nc = opts.output_nc
        self.continue_train = opts.continue_train
        self.which_epoch = opts.which_epoch

        # if hasattr(opts, 'model_type'): self.model_type = opts.model_type
        if hasattr(opts, 'l2_reg_weight'): self.l2_reg_weight = opts.l2_reg_weight
        if hasattr(opts, 'lr_rate'): self.lr_rate = opts.lr_rate
        if hasattr(opts, 'feature_scale'): self.feature_scale = opts.feature_scale
        if hasattr(opts, 'tensor_dim'): self.tensor_dim = opts.tensor_dim
        if hasattr(opts, 'path_pretrained_model'): self.path_pretrained_model = opts.path_pretrained_model
        if hasattr(opts, 'criterion'): self.criterion = opts.criterion


def get_model(json_model_opts):

    model = None
    model_opts = ModelOpts()
    model_opts.initialise(json_model_opts)

    print('\nInitilization model {}'.format(model_opts.model_type))

    model_type = model_opts.model_type
    if model_type == 'seg':
        from .segmentation_model import SegmentationModel
        model = SegmentationModel()
    if model_type == 'det':
        from .maskrcnn_model import MaskrcnnDetection
        model = MaskrcnnDetection()
    elif model_type == 'classifier':
        raise NotImplemented
    elif model_type == 'detection':
        raise NotImplemented

    model.initialize(model_opts)
    print("Model [%s] is initialized" % (model.name()))

    return model



