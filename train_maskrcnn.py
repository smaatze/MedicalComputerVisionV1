
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

from utils.util import json_file_to_pyobj
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transfomation
from models import get_model
from models.utils import train_one_epoch, evaluate


def train(arguments):
    json_filename = arguments.config
    network_debug = arguments.debug

    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    proj_name = train_opts.proj_name

    ds_class = get_dataset(proj_name)
    ds_path = get_dataset_path(proj_name, json_opts.data)
    ds_transform = get_dataset_transfomation(proj_name, opts=json_opts.augmentation)

    model = get_model(json_opts.model)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.net.to(device)

    train_dataset = ds_class(ds_path, transforms=True)
    test_dataset = ds_class(ds_path, transforms=False)

    # split the dataset in train and test set
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-50])
    test_dataset = torch.utils.data.Subset(test_dataset, indices[-50:])

    train_loader = DataLoader(dataset=train_dataset, num_workers=4,
                              batch_size=train_opts.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, num_workers=4,
                             batch_size=1, shuffle=False)

    # construct an optimizer
    params = [p for p in model.net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, test_loader, device=device)

    print("That's it!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CNN Segmentation Training")

    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-d', '--debug', help='debug mode', action='store_true')

    sys.argv = ['train_segmentation.py', '--config', 'config\\config_pennfudan_maskrcnn.json', '--debug']
    # sys.argv = ['train_segmentation.py', '--config', '.config\\config_needle_track_unet.json']

    args = parser.parse_args()
    train(args)