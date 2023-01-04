import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset, DataLoader

# handle PIL errors
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class dataset_sinkhole(Dataset):
    """ Implements a dataset that returns different inputs and their corresponding sinkhole labels """

    def __init__(self,
                 mode,
                 image,
                 dem,
                 dem_max,
                 dem_min,
                 NAIP,
                 label,
                 dem_dx,
                 dem_dy,
                 cutout_size=(400, 400),
                 full_dem_dx=None,
                 full_dem_dy=None,
                 dem_dxy_pre=None):

        self.mode = mode
        self.full_image = image
        self.full_dem = dem
        self.dem_max = dem_max
        self.dem_min = dem_min

        if self.mode != 'train':
            self.full_dem_dx = full_dem_dx
            self.full_dem_dy = full_dem_dy

        self.dem_dx = dem_dx
        self.dem_dy = dem_dy

        self.dem_dxy_pre = dem_dxy_pre

        self.full_NAIP = NAIP
        self.full_label = label
        self.cutout_size = cutout_size

        self.to_tensor = transforms.ToTensor()

        # Normalization methods and values
        #from drive.MyDrive.ML_dataset.config import cfg
        from config import cfg
        self.normalization_shaded = cfg.data.normalize_shaded
        self.normalization_dem = cfg.data.normalize_dem
        self.normalize_dem_ddxy = cfg.data.normalize_dem_ddxy
        self.normalize_dem_pre = cfg.data.normalize_dem_dxy_pre
        self.normalization_naip = cfg.data.normalize_naip

        self.normalize_shaded = transforms.Normalize(
            mean=[0.69829104, 0.38062648, 0.21748482],
            std=[0.0729226, 0.18631184, 0.20916126])

        self.normalize_naip = transforms.Normalize(
            mean=[0.46395134, 0.52389686, 0.38538468, 0.57477817],
            std=[0.14210995, 0.13012627, 0.10734724, 0.15983065])

        self.eval_pad = cfg.data.eval_pad  # padding needed for evaluations

    def __len__(self):
        if self.mode == 'train':
            # ~ 23x28 bins for image size of 9200x11257 and cutout size of 400x400
            return 644
        elif self.mode == 'val':
            # 23x7 bins for an image size of 9200x2800 and cutout size of 400x400
            return 161
        elif self.mode == 'test':
            # 24x35 bins for size 18850 x 14267
            return 840

    def __getitem__(self, idx):
        if self.mode == 'train':
            # during training, apply random cropping

            # get crop parameters
            i, j, h, w = transforms.RandomCrop.get_params(
                self.full_image, output_size=self.cutout_size)

            # apply crop on the shaded releif
            image = transforms_function.crop(self.full_image, i, j, h, w)

            # dem
            dem = transforms_function.crop(self.full_dem, i, j, h, w)
            dem.load()

            dem_dxy_pre = transforms_function.crop(self.dem_dxy_pre, i, j, h, w)
            dem_dxy_pre.load()

            # dem derivatives
            dem_dx = self.dem_dx[i:i + h, j:j + w]
            dem_dy = self.dem_dy[i:i + h, j:j + w]

            # apply crop on the NAIP image
            naip = transforms_function.crop(self.full_NAIP, i, j, h, w)

            # apply crop on the label
            label = transforms_function.crop(self.full_label, i, j, h, w)

        else:
            # test data: # sequentially go through all the data
            num_columns = 7 if self.mode == 'val' else 35
            row, col = divmod(idx, num_columns)

            if self.eval_pad:
                left = max(col * self.cutout_size[0] - 40, 0)
                upper = max(row * self.cutout_size[1] - 40, 0)
                right = (col + 1) * self.cutout_size[0] + 40
                lower = (row + 1) * self.cutout_size[1] + 40

            else:
                left = col * self.cutout_size[0]
                upper = row * self.cutout_size[1]
                right = (col + 1) * self.cutout_size[0]
                lower = (row + 1) * self.cutout_size[1]

            # apply crop on all the inputs
            image = self.full_image.crop((left, upper, right, lower))
            image.load()

            dem = self.full_dem.crop((left, upper, right, lower))
            dem.load()

            dem_dxy_pre = self.dem_dxy_pre.crop((left, upper, right, lower))
            dem_dxy_pre.load()

            dem_dx = self.dem_dx[upper:lower, left:right]
            dem_dy = self.dem_dy[upper:lower, left:right]

            naip = self.full_NAIP.crop((left, upper, right, lower))

            label = self.full_label.crop((left, upper, right, lower))
            label.load()

        # convert to tensor
        image = self.to_tensor(image).float()

        # Normalize shaded relief
        if self.normalization_shaded == 'unit_gaussian':
            image = self.normalize_shaded(image)
        elif self.normalization_shaded == 'instance':
            min_now = torch.min(image)
            max_now = torch.max(image)
            image = (image - min_now) / (max_now - min_now + 1e-7)
        elif self.normalization_shaded == '0_to_1':
            # to_tensor above already converts it to the [0, 1] range
            pass

        # Normalize DEM
        dem_np = np.array(dem).astype(np.float32)

        if self.normalization_dem == 'unit_gaussian':
            dem_np = (dem_np - self.dem_min) / (self.dem_max - self.dem_min)
            dem_np = (dem_np - 0.68867725) / (0.12318235)
        elif self.normalization_dem == '0_to_1':
            dem_np = (dem_np - self.dem_min) / (self.dem_max - self.dem_min
                                                )  # [0, 1]
            # pass # already normalized this way
        elif self.normalization_dem == 'instance':
            min_now = np.min(dem_np)
            max_now = np.max(dem_np)
            dem_np = (dem_np - min_now) / (max_now - min_now + 1e-7)

        elif self.normalization_dem == 'none':
            # no normalization needed
            pass

        # normalize DEM derivatives
        if self.normalize_dem_ddxy == 'instance':
            min_now = np.min(dem_dx)
            max_now = np.max(dem_dx)
            dem_dx = (dem_dx - min_now) / (max_now - min_now + 1e-7)

            min_now = np.min(dem_dy)
            max_now = np.max(dem_dy)
            dem_dy = (dem_dy - min_now) / (max_now - min_now + 1e-7)

        elif self.normalize_dem_ddxy == '0_to_1':
            dem_dx_min = -55.328857
            dem_dx_max = 26.779816
            dem_dx = (dem_dx - dem_dx_min) / (dem_dx_max - dem_dx_min)

            dem_dy_min = -27.779816
            dem_dy_max = 45.826863
            dem_dy = (dem_dy - dem_dy_min) / (dem_dy_max - dem_dy_min)

        elif self.normalize_dem_ddxy == 'unit_gaussian':
            dem_dx_mean = -0.0093376
            dem_dx_std = 0.41069648
            dem_dx = (dem_dx - dem_dx_mean) / dem_dx_std

            dem_dy_mean = -0.0038596862
            dem_dy_std = 0.41352344
            dem_dy = (dem_dy - dem_dy_mean) / dem_dy_std

        # normalize dem derivative: this is pre-computed, conventional slope
        dem_dxy_pre = np.array(dem_dxy_pre).astype(np.float32)

        if self.normalize_dem_pre == 'instance':
            min_now = np.min(dem_dxy_pre)
            max_now = np.max(dem_dxy_pre)
            dem_dxy_pre = (dem_dxy_pre - min_now) / (max_now - min_now + 1e-7)
        elif self.normalize_dem_pre == 'unit_gaussian':
            dem_dxy_pre = (dem_dxy_pre - 4.509918) / 4.211598

        elif self.normalize_dem_pre == '0_to_1':
            dem_dxy_pre = (dem_dxy_pre - 0) / 84.89005  # min=0, max=84.89005

        # normalize naip
        naip = self.to_tensor(naip).float()
        if self.normalization_naip == 'unit_gaussian':
            # print('naip shape: ', image.shape)
            naip = self.normalize_naip(naip)
        elif self.normalization_naip == 'instance':
            min_now = torch.min(naip)
            max_now = torch.max(naip)
            naip = (naip - min_now) / (max_now - min_now + 1e-7)
        elif self.normalization_naip == '0_to_1':
            pass

        # convert dem and naip to tensor
        dem_tensor = torch.from_numpy(dem_np).float()
        dem_dxy_pre_tensor = torch.from_numpy(dem_dxy_pre).float()

        label = torch.from_numpy(np.array(label))

        dem_dx = torch.from_numpy(dem_dx).unsqueeze(0)
        dem_dy = torch.from_numpy(dem_dy).unsqueeze(0)
        dem_dxy = torch.cat([dem_dx, dem_dy], dim=0)

        # New added
        # combine dem and its varients with naip
        image_naip = torch.cat([image, naip], dim=0)
        dem_naip = torch.cat([dem_tensor.unsqueeze(0), naip], dim=0)
        # print(dem_dxy.shape)
        # print(naip.shape)
        if self.mode != 'train':
            diff = max(dem_dxy.shape[1], naip.shape[1]) - min(dem_dxy.shape[1], naip.shape[1])
            dem_dxy_naip = torch.cat([nn.ZeroPad2d((0, 0, 0, diff))(dem_dxy), naip], dim=0)
        else:
            dem_dxy_naip = torch.cat([dem_dxy, naip], dim=0)
        dem_dxy_pre_naip = torch.cat([dem_dxy_pre_tensor.unsqueeze(0), naip], dim=0)

        # print('image size:', image.shape)
        # print('dem_tensor size:', dem_tensor.shape)
        # print('naip size:', naip.shape)
        # print('label size:', label.shape)
        # print('dem_dxy size:', dem_dxy.shape)
        # print('dem_dxy_pre_tensor', dem_dxy_pre_tensor.shape)
        # print(dem_dxy_naip.shape)

        return image, dem_tensor, naip, label, idx, dem_dxy, dem_dxy_pre_tensor, \
            image_naip, dem_naip, dem_dxy_naip, dem_dxy_pre_naip


def get_data(cfg):
    """ Reads data from the disk, computes statistics, and return train/val/test dataloaders  """
    data_dir = cfg.data.data_dir
    """ Read images and print details """

    # shaded relief
    image_name = os.path.join(data_dir, 'ShadedRelief_Raster.tif')
    image_full = Image.open(image_name)
    print('shaded relief size: ', image_full.size)
    print('shaded relief maximum: ', max(image_full.getdata()))
    print('shaded relief minimum: ', min(image_full.getdata()))

    # DEM
    dem_name = os.path.join(data_dir, 'Ky_DEM_KYAPED_5FT_3.tif')
    dem_full = Image.open(dem_name)

    # save the max and min of dem for [0, 1] normalization
    dem_max = max(dem_full.getdata())
    dem_min = min(dem_full.getdata())

    print('DEM size: ', dem_full.size)
    print('DEM maximum: ', dem_max)
    print('DEM minimum: ', dem_min)

    dem_dx = np.load(os.path.join(data_dir, 'dem_dx.npy'))
    dem_dy = np.load(os.path.join(data_dir, 'dem_dy.npy'))
    print('DEM X-Derivative size: ', dem_dx.shape)
    print('DEM Y-Derivative size: ', dem_dy.shape)

    # Native, pre-computed derivatives
    dem_dxy_pre_name = os.path.join(data_dir, 'Slope_Ky_DEM_KYAPED_5FT_3.tif')
    dem_dxy_pre = Image.open(dem_dxy_pre_name)

    NAIP_name = os.path.join(data_dir, 'Ky_NAIP_2018_5FT.tif')
    NAIP_full = Image.open(NAIP_name)
    print('NAIP size: ', NAIP_full.size)
    print('NAIP maximum: ', max(NAIP_full.getdata()))
    print('NAIP minimum: ', min(NAIP_full.getdata()))

    # load annotation
    label_name = os.path.join(data_dir, 'SinkholeBinaryRaster.tif')
    label_full = Image.open(label_name)
    print('label size: ', label_full.size)
    print('label maximum: ', max(label_full.getdata()))
    print('label minimum: ', min(label_full.getdata()))
    """ Make splits """
    split_ratio = 0.8
    print('image width: ', image_full.size[0])
    train_width = int(0.8 * image_full.size[0])
    train_val_height = 9229  # this is to match the previous tiles

    print('training: ', 100 * split_ratio, '% width = ', train_width)

    # Training images
    # shaded relief
    image_train = image_full.crop((0, 0, train_width, train_val_height))
    image_train.load()

    # DEM
    dem_train = dem_full.crop((0, 0, train_width, train_val_height))
    dem_train.load()
    print('dem train size:', dem_train.size)

    # pre-compute DEM derivative
    dem_dxy_train_pre = dem_dxy_pre.crop((0, 0, train_width, train_val_height))
    dem_dxy_train_pre.load()
    print('train dem derivative precomputed:', dem_dxy_train_pre.size)

    dem_dx_train = dem_dx[:train_val_height, :train_width]
    dem_dy_train = dem_dy[:train_val_height, :train_width]

    print('train dem dxy train shape:', dem_dx_train.shape)

    # NAIP image
    NAIP_train = NAIP_full.crop((0, 0, train_width, train_val_height))
    NAIP_train.load()

    label_train = label_full.crop((0, 0, train_width, train_val_height))
    label_train.load()
    print('label train size:', label_train.size)

    # Val images
    # shaded relief
    image_val = image_full.crop(
        (train_width, 0, image_full.size[0], train_val_height))
    image_val.load()

    # DEM
    dem_val = dem_full.crop(
        (train_width, 0, image_full.size[0], train_val_height))
    dem_val.load()
    print('val dem val size:', dem_val.size)

    dem_dxy_val_pre = dem_dxy_pre.crop(
        (train_width, 0, image_full.size[0], train_val_height))
    dem_dxy_val_pre.load()
    print('val dem derivative precomputed:', dem_dxy_val_pre.size)

    dem_dx_val = dem_dx[0:train_val_height, train_width:]
    dem_dy_val = dem_dy[0:train_val_height, train_width:]

    print('val dem dxy val shape:', dem_dy_val.shape)

    # NAIP image
    NAIP_val = NAIP_full.crop(
        (train_width, 0, image_full.size[0], train_val_height))
    NAIP_val.load()

    label_val = label_full.crop(
        (train_width, 0, image_full.size[0], train_val_height))
    label_val.load()
    print('label val size:', label_val.size)

    # Test images
    # shaded relief
    image_test = image_full.crop(
        (0, train_val_height, image_full.size[0], image_full.size[1]))
    image_test.load()

    # DEM
    dem_test = dem_full.crop(
        (0, train_val_height, image_full.size[0], image_full.size[1]))
    dem_test.load()
    print('test dem test size:', dem_test.size)

    dem_dxy_test_pre = dem_dxy_pre.crop(
        (0, train_val_height, image_full.size[0], image_full.size[1]))
    dem_dxy_test_pre.load()
    print('test dem derivative precomputed:', dem_dxy_test_pre.size)

    dem_dx_test = dem_dx[train_val_height:, :]
    dem_dy_test = dem_dy[train_val_height:, :]

    print('test dem dxy test shape:', dem_dx_test.shape)

    # NAIP image
    NAIP_test = NAIP_full.crop(
        (0, train_val_height, image_full.size[0], image_full.size[1]))
    NAIP_test.load()

    label_test = label_full.crop(
        (0, train_val_height, image_full.size[0], image_full.size[1]))
    label_test.load()
    print('label test size:', label_test.size)

    # instantiate dataset classes
    train_dataset = dataset_sinkhole(mode='train',
                                     image=image_train,
                                     dem=dem_train,
                                     dem_max=dem_max,
                                     dem_min=dem_min,
                                     NAIP=NAIP_train,
                                     label=label_train,
                                     cutout_size=cfg.data.cutout_size,
                                     dem_dx=dem_dx_train,
                                     dem_dy=dem_dy_train,
                                     dem_dxy_pre=dem_dxy_train_pre)

    # added for debugging
    #train_dataset[0]

    val_dataset = dataset_sinkhole(mode='val',
                                   image=image_val,
                                   dem=dem_val,
                                   dem_max=dem_max,
                                   dem_min=dem_min,
                                   NAIP=NAIP_test,
                                   label=label_val,
                                   cutout_size=cfg.data.cutout_size,
                                   dem_dx=dem_dx_val,
                                   dem_dy=dem_dy_val,
                                   full_dem_dx=dem_dx,
                                   full_dem_dy=dem_dy,
                                   dem_dxy_pre=dem_dxy_val_pre)

    test_dataset = dataset_sinkhole(mode='test',
                                    image=image_test,
                                    dem=dem_test,
                                    dem_max=dem_max,
                                    dem_min=dem_min,
                                    NAIP=NAIP_test,
                                    label=label_test,
                                    cutout_size=cfg.data.cutout_size,
                                    dem_dx=dem_dx_test,
                                    dem_dy=dem_dy_test,
                                    full_dem_dx=dem_dx,
                                    full_dem_dy=dem_dy,
                                    dem_dxy_pre=dem_dxy_test_pre)

    # prepare dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              num_workers=cfg.train.num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=cfg.train.shuffle,
                            num_workers=cfg.train.num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.train.batch_size,
                             shuffle=cfg.train.shuffle,
                             num_workers=cfg.train.num_workers)

    return train_loader, val_loader, test_loader
