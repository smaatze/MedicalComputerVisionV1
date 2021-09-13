
import os
import sys
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transfomation
from utils.util import json_file_to_pyobj
from models import get_model
# from models.base_model import get_n_parameters
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger


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
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        # print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        # exit()

    train_dataset = ds_class(ds_path, split='train', transform=ds_transform['train'],
                             preload_data=train_opts.preload_data)
    valid_dataset = ds_class(ds_path, split='valid', transform=ds_transform['valid'],
                             preload_data=train_opts.preload_data)
    test_dataset  = ds_class(ds_path, split='test', transform=ds_transform['valid'],
                             preload_data=train_opts.preload_data)
    train_loader  = DataLoader(dataset=train_dataset, nm_workers=16,
                              batch_size=train_opts.batch_size, shuffle=True)
    valid_loader  = DataLoader(dataset=valid_dataset, nm_workers=16,
                              batch_size=train_opts.batch_size, shuffle=False)
    test_loader   = DataLoader(dataset=test_dataset, nm_workers=16,
                              batch_size=train_opts.batch_size, shuffle=False)

    visualiser = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iter: %d)' % (epoch, len(train_loader)))

        # trainig iteration
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader,1), total=len(train_loader)):
            # make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            # model.optimize_parameters_accumulate_grd(epoch_iter)

            # error visualization
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # validation and testing iteration
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels) in tqdm(enumerate(loader,1), total=len(loader)):
                model.set_input(images, labels)
                model.validate()

                # error visualization
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                # visualize predictions
                visuals = model.get_current_visuals()
                visualiser.display_current_results(visuals)

        # update the plots
        for split in ['train', 'validation', 'test']:
            visualiser.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualiser.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)

        error_logger.reset()

        # save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # update the model learning rate
        model.update_learning_rate()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CNN Segmentation Training")

    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-d', '--debug', help='debug mode', action='store_true')

    sys.argv = ['train_segmentation.py', '--config', 'config\\config_needle_track_unet.json', '--debug']
    # sys.argv = ['train_segmentation.py', '--config', '.config\\config_needle_track_unet.json']

    args = parser.parse_args()
    train(args)