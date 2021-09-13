
import sys
import math
import time
import numpy as np
import torchvision
import torch.optim as optim
from .losses import *
from utils.metrics import segmentation_scores, dice_score_list
from .pytorch_utils import MetricLogger, SmoothedValue, \
    warmup_lr_scheduler, reduce_dict, MetricLogger
from .coco_utils import get_coco_api_from_dataset, CocoEvaluator

def get_criterion(opts):
    if opts.criterion == 'cross_entropy':
        if opts.model_type == 'seg':
            criterion = cross_entropy_2d if opts.tensor_dim == '2d' else cross_entropy_3d
    elif opts.criterion == 'dice_loss':
        criterion = SoftDiceLoss(opts.output_nc)
    return criterion

def get_optimizer(option, params):
    opt_alg = 'sgd' if not hasattr(option,'optim') else option.optim
    if opt_alg == 'sgd':
        optimizer = optim.SGD(params,
                              lr=option.lr_rate,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay = option.l2_reg_weight)

    if opt_alg == 'adam':
        optimizer = optim.Adam(params,
                              lr=option.lr,
                              betas=(0.9, 0.999),
                              weight_decay=option.l2_reg_weight)

    return optimizer

def apply_argmax_softmax(pred):
    log_p = F.softmax(pred, dim=1)
    return log_p

def segmentation_stats(pred_seg, target):
    n_classes = pred_seg.size(1)
    pred_lbls = pred_seg.data.max(1)[1].cpu().numpy()
    gt = np.squeeze(target.data.cpu().numpy(), axis=1)
    gts, preds = [], []
    for gt_, pred_ in zip(gt, pred_lbls):
        gts.append(gt_)
        preds.append(pred_)

    iou = segmentation_scores(gts, preds, n_class=n_classes)
    dice = dice_score_list(gts, preds, n_class=n_classes)

    return iou, dice

def get_n_parameters(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params

# returns the fp/bp times of the model
def get_fp_bp_time (self, size=None):
    if size is None:
        size = (1, 1, 160, 160, 96)

    inp_array = Variable(torch.zeros(*size)).cuda()
    out_array = Variable(torch.zeros(*size)).cuda()
    fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

    bsize = size[0]
    return fp/float(bsize), bp/float(bsize)


def measure_fp_bp_time(model, x, y):
    # synchronize gpu time and measure fp
    torch.cuda.synchronize()
    t0 = time.time()
    y_pred = model(x)
    torch.cuda.synchronize()
    elapsed_fp = time.time() - t0

    if isinstance(y_pred, tuple):
        y_pred = sum(y_p.sum() for y_p in y_pred)
    else:
        y_pred = y_pred.sum()

    # zero gradients, synchronize time and measure
    model.zero_grad()
    t0 = time.time()
    # y_pred.backward(y)
    y_pred.backward()
    torch.cuda.synchronize()
    elapsed_bp = time.time() - t0
    return elapsed_fp, elapsed_bp


def benchmark_fp_bp_time(model, x, y, n_trial=1000):
    # transfer the model on GPU
    model.cuda()

    # DRY RUNS
    for i in range(10):
        _, _ = measure_fp_bp_time(model, x, y)

    print('DONE WITH DRY RUNS, NOW BENCHMARKING')

    # START BENCHMARKING
    t_forward = []
    t_backward = []

    print('trial: {}'.format(n_trial))
    for i in range(n_trial):
        t_fp, t_bp = measure_fp_bp_time(model, x, y)
        t_forward.append(t_fp)
        t_backward.append(t_bp)

    # free memory
    del model

    return np.mean(t_forward), np.mean(t_backward)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator