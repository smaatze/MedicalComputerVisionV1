from unittest.mock import Base

from .base_model import BaseModel
from torch.autograd import Variable
from collections import OrderedDict

from .base_model import BaseModel
from .networks import get_network
from .utils import get_criterion, get_optimizer, apply_argmax_softmax
import utils.util as util
from .networks.utils import print_network, get_scheduler
from models.utils import segmentation_stats

class MaskrcnnDetection(BaseModel):

    def name(self):
        return 'MaskRcnnDetection'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)

        self.is_train = opts.is_train
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim

        self.net = get_network(opts.model_name, n_classes=opts.output_nc)

        if self.use_cuda: self.net.cuda()

        if not self.is_train or opts.continue_train:
            self.path_pretrain_model = opts.path_pretrain_model
            if self.path_pretrain_model:
                self.load_network_from_path(self.net, self.path_pretrain_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.is_train:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_s = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_s)

        # print the network detail
        if kwargs.get('verbose', True):
            print('network in initialized')
            print_network(self.net)

    def set_scheduler(self, train_opts):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opts))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    def set_input(self, *inputs):
        for idx, _input in enumerate(inputs):
            bs = _input.size()
            if (self.tensor_dim == '2d') and (len(bs) > 4):
                _input = _input.permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                assert self.input.size() == self.target.size()

    def forward(self, split):
        if split == 'train':
            self.prediction = self.net(Variable(self.input))
        elif split == 'test':
            self.prediction = self.net(Variable(self.input, volatile=True))
            self.logits = apply_argmax_softmax(self.prediction)
            if __name__ == '__main__':
                self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    def backward(self):
        self.loss_S = self.criterion(self.prediction, self.target)
        self.loss_S.backward()

    def optimize_parameters(self):
        self.net.train()
        self.forward()
        self.optimizer_s.step()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)

    def get_segmentation_stats(self):
        self.seg_scores, self.dice_score = segmentation_stats()

    def get_current_errors(self):
        return OrderedDict([('Seg_loss', self.loss_S.data[0])])

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input, 'img')
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)
