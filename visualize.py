# This code initially performed both evaluation and visualization. The evaluation code has now been moved to eval_threshold. To avoid confusion, I am not renaming this file for now

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from model import Unet, Unet_early
from fusenet import FuseNet
from data_factory import get_data
from config import cfg


def visualize():
    """
    Loads a saved checkpoint from disk, and saves qualitative results on the disk.
    Note that this script saves cutout examples -- full-size outputs are generated by eval.py
    """

    # model type: 'best' (based on lowest val loss) or 'end'
    eval_mode = 'best'

    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        raise ValueError(
            'The directory with trained model does not exist! Make sure cfg.train.out_dir in config.py has the correct directory name'
        )

    if cfg.model.name == 'unet':
        model = Unet(in_channels=cfg.data.input_channels,
                     out_channels=2,
                     feature_reduction=4,
                     norm_type=cfg.model.norm_type)

    elif cfg.model.name == 'unet_early':
        model = Unet_early(in_channels=cfg.data.input_channels,
                           out_channels=2,
                           feature_reduction=4,
                           norm_type=cfg.model.norm_type)

    elif cfg.model.name == 'fusenet':
        # change feature reduction to 1 if use pre-trained model
        if cfg.model.pre_trained:
            model = FuseNet(in_channels=cfg.data.input_channels,
                            out_channels=2,
                            feature_reduction=1,
                            norm_type=cfg.model.norm_type)
        else:
            model = FuseNet(in_channels=cfg.data.input_channels,
                            out_channels=2,
                            feature_reduction=4,
                            norm_type=cfg.model.norm_type)

    # which checkpoint to load: best (lowest val loss) or the one saved at the end of training
    if eval_mode == 'best':
        fname = os.path.join(out_dir, 'model_dict.pth')
    else:
        fname = os.path.join(out_dir, 'model_dict_end.pth')

    model.load_state_dict(torch.load(fname))
    model.eval()

    # get dataloader
    data_mode = 'test'
    cfg.train.batch_size = 1
    # enable padding in the evaluation
    cfg.data.eval_pad = True
    cfg.train.shuffle = False
    _, data_loader_val, data_loader_test = get_data(cfg)

    if data_mode == 'val':
        data_loader = data_loader_val
    elif data_mode == 'test':
        data_loader = data_loader_test

    try:
        fname = os.path.join(out_dir, 'best_threshold.txt')
        with open(fname, 'r') as f:
            threshold = float(f.read())
        print('loaded threshold saved by the eval script; threshold=',
              threshold)
    except:
        threshold = 0.9
        print(
            'Could not locate threshold saved by the eval script; threshold=0.9'
        )

    ctr = 0
    save_dir = os.path.join(out_dir, 'cutout_results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    iterator = iter(data_loader)
    n_cutouts = 100
    for i in range(n_cutouts):
        data = next(iterator)
        shaded = data[0].cuda()
        dem = data[1].cuda().unsqueeze(1)
        naip_image = data[2].cuda()
        labels = data[3].long().cuda()
        dem_dxy = data[5].cuda()
        dem_dxy_pre = data[6].cuda().unsqueeze(1)

        predictions = model(shaded, dem, naip_image, dem_dxy, dem_dxy_pre)
        predictions = torch.softmax(predictions, dim=1)

        if predictions.shape[2] == 420:
            starty = 0
            endy = 400
        else:
            starty = 40
            endy = 440

        if predictions.shape[3] == 420:
            startx = 0
            endx = 400
        else:
            startx = 40
            endx = 440

        predictions = predictions[:, :, starty:endy, startx:endx]
        dem = dem[:, :, starty:endy, startx:endx]
        labels = labels[:, starty:endy, startx:endx]

        pred_final = (predictions[:, 1, :, :] > threshold).long()

        k = 0
        dem_min, dem_max = torch.min(dem[k, 0, :, :]), torch.max(dem[k,
                                                                     0, :, :])
        dem_instance_norm = ((dem[k, 0, :, :] - dem_min) /
                             (dem_max - dem_min)).detach().cpu().numpy()

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 4, 1)
        plt.imshow(dem_instance_norm, vmin=0, vmax=1, cmap='Greens')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(predictions[k, 1, :].detach().cpu().numpy(),
                   vmin=0,
                   vmax=1,
                   cmap='Blues')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(pred_final[k, :, :].detach().cpu().numpy(),
                   vmin=0,
                   vmax=1,
                   cmap='Blues')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(labels[k, :, :].detach().cpu().numpy(),
                   vmin=0,
                   vmax=1,
                   cmap='Blues')
        plt.axis('off')
        name_str = str(ctr) + '_result.jpg'
        fname = os.path.join(save_dir, name_str)
        plt.savefig(fname, pad_inches=0, bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.imshow(dem_instance_norm, vmin=0, vmax=1, cmap='Greens')
        name_str = str(ctr) + '_dem.jpg'
        fname = os.path.join(save_dir, name_str)
        plt.axis('off')
        plt.savefig(fname, pad_inches=0, bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.imshow(shaded[k,].permute(1, 2, 0).detach().cpu().numpy())
        name_str = str(ctr) + '_shaded.jpg'
        fname = os.path.join(save_dir, name_str)
        plt.savefig(fname, pad_inches=0, bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.imshow(predictions[k, 1, :].detach().cpu().numpy(),
                   vmin=0,
                   vmax=1,
                   cmap='Blues')
        plt.axis('off')
        name_str = str(ctr) + '_soft_pred.jpg'
        fname = os.path.join(save_dir, name_str)
        plt.savefig(fname, pad_inches=0, bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.imshow(pred_final[k, :, :].detach().cpu().numpy(),
                   vmin=0,
                   vmax=1,
                   cmap='Blues')
        plt.axis('off')
        name_str = str(ctr) + '_binary_pred.jpg'
        fname = os.path.join(save_dir, name_str)
        plt.savefig(fname, pad_inches=0, bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.imshow(labels[k, :, :].detach().cpu().numpy(),
                   vmin=0,
                   vmax=1,
                   cmap='Blues')
        #plt.title('GT label')
        plt.axis('off')
        name_str = str(ctr) + '_labels.jpg'
        fname = os.path.join(save_dir, name_str)
        plt.savefig(fname, pad_inches=0, bbox_inches='tight')
        plt.close()

        ctr += 1

    print('finished saving cutout results')


if __name__ == '__main__':
    visualize()
