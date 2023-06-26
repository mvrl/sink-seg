'''
The implementation of Unsupervised Adapation method
from paper Unsupervised Adaptation of Semantic Segmentation
Models without Source Data: https://arxiv.org/abs/2112.02359
'''

import random
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from model import Unet
from config import cfg

import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
import torchvision.ops as ops
from torch.utils.data import Dataset, DataLoader
from evaluate import get_confusion_matrix_binary

# handle PIL errors
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def set_parameter_requires_grad(model, feature_extracting):
    # we do not need gradients for feature extraction
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# define dataset class for adaptive learning
class dataset_adaptive(Dataset):

    def __init__(self,
                 mode,
                 dem_dx,
                 dem_dy,
                 label,
                 cutout_size=(400, 400)):
        
        self.mode = mode
        self.dem_dx = dem_dx
        self.dem_dy = dem_dy
        self.label = label
        self.cutout_size = cutout_size
    
        self.normalize_dem_ddxy = cfg.data.normalize_dem_ddxy

    def __len__(self):

        # if self.mode == 'train':
        #     return 7*15
        # elif self.mode == 'val':
        #     return 7*15
        # elif self.mode == 'test':
        #     return 14*15
        return 14*15


    def __getitem__(self, idx):

        #if self.mode == 'train':

        # concat
        dem_dx = torch.from_numpy(self.dem_dx).unsqueeze(0)
        dem_dy = torch.from_numpy(self.dem_dy).unsqueeze(0)
        dem_dxy = torch.cat([dem_dx, dem_dy], dim=0)

        label = torch.from_numpy(self.label)

        # data normalization
        if self.normalize_dem_ddxy == 'unit_gaussian':
            dem_dx_mean = -0.0024001286
            dem_dx_std = 0.13208884
            dem_dx = (dem_dx - dem_dx_mean) / dem_dx_std
            dem_dxy[0] = dem_dx
 
            dem_dy_mean = -0.00279918
            dem_dy_std = 0.12848537
            dem_dy = (dem_dy - dem_dy_mean) / dem_dy_std
            dem_dxy[1] = dem_dy
        
        if self.mode == 'train':
            # random cropping
            i, j, h, w = transforms.RandomCrop.get_params(
                dem_dxy, output_size=self.cutout_size)

            dem_dxy = dem_dxy[:, i:i + h, j:j + w]
            label = label[i:i + h, j:j + w]
        else:
            num_columns = 15
            row, col = divmod(idx, num_columns)
            
            left = col * self.cutout_size[0]
            upper = row * self.cutout_size[1]
            right = (col + 1) * self.cutout_size[0]
            lower = (row + 1) * self.cutout_size[1]

            dem_dxy = dem_dxy[:, upper:lower, left:right]
            label = label[upper:lower, left:right]

        return dem_dxy, label

# Mirror image
class Mirror(nn.Module):
    def __init__(self):
        super(Mirror, self).__init__()
    
    def forward(self, img):
        
        width = cfg.data.cutout_size[1]
        # column to split
        col = random.randint(50, width-50)

        dim = len(img.shape)
        
        if col < int(width/2):
            if dim == 4:
                half_img = img[:, :, :, col:]
            else:
                half_img = img[:, :, col:]
        else:
            if dim == 4:
                half_img = img[:, :, :, :col]
            else:
                half_img = img[:, :, :col]
        
        flip_half = transforms_function.hflip(half_img)
        
        new_img = torch.cat([flip_half, half_img], dim=dim-1) \
        if col < int(width/2) else torch.cat([half_img, flip_half], dim=dim-1)
        new_img = transforms_function.resize(new_img, cfg.data.cutout_size, 
                                            antialias=True
        )

        return new_img


# Define loss
class AdaptLoss(nn.Module):
    def __init__(self, weight):
        super(AdaptLoss, self).__init__()

        self.weight = weight
        # classification loss
        self.Lc = nn.CrossEntropyLoss(weight=self.weight)
        self.Lr1 = nn.CrossEntropyLoss(weight=self.weight)
        self.Lr2 = nn.CrossEntropyLoss(weight=self.weight)
    
    def forward(self, data):
        loss = self.Lc(data[0], data[1]) + self.Lr1(data[2], data[3]) + \
                self.Lr2(data[2], data[4])
        return loss

def get_data():
    # Read data
    data_dir = cfg.data.data_dir
    dem_dx = np.load(os.path.join(data_dir, 'B3_dem_dx.npy'))
    dem_dy = np.load(os.path.join(data_dir, 'B3_dem_dy.npy'))
    print('DEM X-Derivative size: ', dem_dx.shape)
    print('DEM Y-Derivative size: ', dem_dy.shape)

    #train_width = int(0.8*dem_dx.shape[1])
    train_val_height = int(0.5*dem_dx.shape[0])

    dem_dx_train = dem_dx[:train_val_height, :]
    dem_dy_train = dem_dy[:train_val_height, :]

    dem_dx_val = dem_dx[train_val_height:, :]
    dem_dy_val = dem_dy[train_val_height:, :]

    # dem_dx_test = dem_dx[train_val_height:, :]
    # dem_dy_test = dem_dy[train_val_height:, :]

    label_name = os.path.join(data_dir, 'sinkhole_binary_5ft_GreeneCoData.tif')
    label_full = Image.open(label_name)
    label_full = np.array(label_full)
    print('label size: ', label_full.shape)

    label_train = label_full[:train_val_height, :]
    print('label train size:', label_train.shape)

    label_val = label_full[train_val_height:, :]
    print('label val size:', label_val.shape)

    train_dataset = dataset_adaptive('train', dem_dx, dem_dy,
                                label_full, cutout_size=cfg.data.cutout_size)
    
    val_dataset = dataset_adaptive('val', dem_dx_val, dem_dy_val, label_val,
                                cutout_size=cfg.data.cutout_size)
    
    test_dataset = dataset_adaptive('test', dem_dx, dem_dy, label_full,
                                cutout_size=cfg.data.cutout_size)
    
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=True,
                            num_workers=cfg.train.num_workers)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=cfg.train.num_workers)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=cfg.train.num_workers)
    
    return train_loader, val_loader, test_loader

def adaptive_train():

    # select device
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")

    # the source model in paper
    model_s = Unet(in_channels=cfg.data.input_channels,
                     out_channels=2,
                     feature_reduction=4,
                     norm_type=cfg.model.norm_type)
    model_s.to(device)

    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        raise ValueError(
            'The directory with trained model does not exist! Make sure cfg.train.out_dir in config.py has the correct directory name'
        )
    fname = os.path.join(out_dir, 'model_dict.pth')

    model_s.load_state_dict(torch.load(fname))
    set_parameter_requires_grad(model_s, feature_extracting=True)

    # the target model in paper
    model_t = Unet(in_channels=cfg.data.input_channels,
                     out_channels=2,
                     feature_reduction=4,
                     norm_type=cfg.model.norm_type)
    model_t.to(device)
    
    train_loader, val_loader, test_loader = get_data()
    
    optim = torch.optim.Adam(model_t.parameters(),
                             lr=cfg.train.learning_rate,
                             weight_decay=cfg.train.l2_reg)
    
    # lr schedular
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=cfg.train.lr_decay_every, gamma=cfg.train.lr_decay)

    weight = torch.tensor([0.05, 1.0]).to(device)
    criterion = AdaptLoss(weight=weight)

    spatial_transform = nn.Sequential(
        transforms.RandomRotation(5),
        Mirror()
    )

    augment_transform = nn.Sequential(
        ops.DropBlock2d(p=0.3, block_size=7),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0))
    )

    val_measure = nn.CrossEntropyLoss(weight=weight)
    best_train_loss = 999.0
    print('starting training')
    for epoch in range(cfg.train.num_epochs):
        loss_train = 0
        cfg.data.mode = 'train'
        model_t.train()
        for i, (X, label) in enumerate(train_loader):
            
            X, _ = X.to(device), label

            optim.zero_grad()
            # M_s(X)
            MsX = model_s(None, None, None, X, None,
                            None, None, None, None)
            
            # pseudo-labeling PL(M_s(X))
            pred = torch.softmax(MsX, dim=1)
            PL_MsX = (pred[:, 1, :, :] > 0.9).long()
            # M_t(X)
            MtX = model_t(None, None, None, X, None,
                            None, None, None, None)
            
            spatial = random.choice(spatial_transform)
            augment = random.choice(augment_transform)
            Ti = nn.Sequential(spatial, augment)

            # M_t(T_i(X))
            MtTiX = model_t(None, None, None, Ti(X), None,
                            None, None, None, None)

            pred = torch.softmax(MtX, dim=1)

            # T_o(M_t(X))
            ToMtX = spatial(pred)
            # T_o(PL(M_t(X)))
            PL_MtX = (pred[:, 1, :, :] > 0.9).long()
            ToPL_MtX = spatial(PL_MtX)

            data = (MtX, PL_MsX, MtTiX, ToMtX, ToPL_MtX)
            loss = criterion(data)

            loss.backward()

            loss_train += loss.item()

            optim.step()
        
        # end of training for this epoch
        loss_train /= len(train_loader)

        # begin validation
        # loss_val = 0
        # model_t.eval()
        # cfg.data.mode = 'eval'
        # with torch.no_grad():
        #     for i, (X, label) in enumerate(val_loader):
        #         optim.zero_grad()  # clear gradients

        #         X, label = X.to(device), label.to(device)

        #         pred = model_t(None, None, None, X, None,
        #                        None, None, None, None)
                
        #         label = label.long()
        #         loss = val_measure(pred, label)
        #         loss_val += loss.item()
        
        # # end of validation
        # loss_val /= len(val_loader)

        # End of epoch
        scheduler.step()

        print('End of epoch ', epoch + 1, ' , Train loss: ', loss_train)
                #', val loss: ', loss_val)

        # save best model checkpoint
        if loss_train < best_train_loss:
            best_train_loss = loss_train
            fname = 'model_t_dict.pth'
            torch.save(model_t.state_dict(), os.path.join(out_dir, fname))
            print('=========== model saved at epoch: ', epoch + 1,
                  ' =================')

    fname = 'model_t_dict_end.pth'
    torch.save(model_t.state_dict(), os.path.join(out_dir, fname))
    print('=========== model saved in the end =============')

    return

def evaluate():

    # select device
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")

    train_loader, val_loader, test_loader = get_data()
    
    # the target model in paper
    model_t = Unet(in_channels=cfg.data.input_channels,
                     out_channels=2,
                     feature_reduction=4,
                     norm_type=cfg.model.norm_type)
    model_t.to(device)
    
    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        raise ValueError(
            'The directory with trained model does not exist! Make sure cfg.train.out_dir in config.py has the correct directory name'
        )
    fname = os.path.join(out_dir, 'model_dict.pth')
    model_t.load_state_dict(torch.load(fname))
    model_t.eval()

    num_classes = 2
    """ Find out best threshold on the val set """
    print('\nFinding out the best threshold on val loader')
    best_iou = torch.zeros(1).to(device)
    best_acc = torch.zeros(1).to(device)
    best_threshold = torch.zeros(1).to(device)
    thresholds = np.linspace(start=0, stop=1, num=21)

    acc_log = []
    iou_log = []
    thresh_log = []

    iou_new = []
    acc_new = []

    for j in range(thresholds.shape[0]):
        threshold = thresholds[j]

        confusion_matrix = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for i, (X, label) in enumerate(test_loader):
                X, label = X.to(device), label.to(device)

                # plt.figure(figsize=(20,8))
                # plt.subplot(1,3,1)
                # plt.imshow(label[0, :400, :400], interpolation='nearest')
                # plt.axis('off')

                # plt.subplot(1,3,2)
                # plt.imshow(X[0, 0, :400, :400])

                # plt.subplot(1,3,3)
                # plt.imshow(X[0, 1, :400, :400])
                # plt.show()

                pred = model_t(None, None, None, X, None,
                                None, None, None, None)
                
                pred = torch.softmax(pred, dim=1)
                
                pred_final = (pred[:, 1, :, :] > threshold).long()

                confusion_matrix += get_confusion_matrix_binary(
                    label=label,
                    pred=pred_final,
                    size=cfg.data.cutout_size,
                    num_class=num_classes)

        # compute metrics from the confusion matrix
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum() / pos.sum()
        mean_acc = (tp / np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        sinkhole_iou = IoU_array[1]

        iou_new.append(IoU_array[1] * 100.0)
        acc_new.append(mean_acc)
        thresh_log.append(threshold)

        print('iou: ', sinkhole_iou, ' threshold: ', threshold)

        if sinkhole_iou > best_iou:
            best_iou = sinkhole_iou
            best_threshold = threshold

    print('==== finished analysis ==== best iou: ', best_iou.item(),
          ' best threshold: ', best_threshold)

    return

if __name__ == '__main__':
    
    adaptive_train()
    evaluate()
