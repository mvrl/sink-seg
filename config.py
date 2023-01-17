# This file contains configuration for training and evaluation

from easydict import EasyDict as edict

cfg = edict()

# Model
cfg.model = edict()
# 'unet', 'fusenet', 'unet_early'
cfg.model.name = 'fusenet'
cfg.model.norm_type = 'batch'  # 'batch', instance
# Fusion type for the model
cfg.model.pre_trained = False

# Data
cfg.data = edict()
cfg.data.data_dir = '../sink-seg-doc/data'
cfg.data.name = 'full_tiles'

# define model input. options: 'dem', 'shaded_relief', 'naip', 'dem_derivative', 'dem_dxy_pre'
# new added: 'shaded_relief_naip', 'dem_naip', 'dem_derivative_naip', 'dem_dxy_pre_naip'
cfg.data.input_type = 'dem_derivative_naip'

if cfg.data.input_type == 'dem':
    cfg.data.input_channels = 1
elif cfg.data.input_type == 'shaded_relief':
    cfg.data.input_channels = 3
elif cfg.data.input_type == 'naip':
    cfg.data.input_channels = 4
elif cfg.data.input_type == 'dem_derivative':
    cfg.data.input_channels = 2
elif cfg.data.input_type == 'dem_dxy_pre':
    cfg.data.input_channels = 1
elif cfg.data.input_type == 'shaded_relief_naip':
    cfg.data.input_channels = 7
elif cfg.data.input_type == 'dem_naip':
    cfg.data.input_channels = 5
elif cfg.data.input_type == 'dem_derivative_naip':
    cfg.data.input_channels = 6
elif cfg.data.input_type == 'dem_dxy_pre_naip':
    cfg.data.input_channels = 5

cfg.data.eval_pad = False

# Normalization of inputs

# DEM normalization. Options: '0_to_1', 'unit_gaussian', 'instance', 'none', '0_to_255'
cfg.data.normalize_dem = '0_to_255'

# shaded relief normalization. Options: '0_to_1', 'unit_gaussian', 'instance'
cfg.data.normalize_shaded = '0_to_1'

# NAIP normalization. Options: '0_to_1', 'unit_gaussian', 'instance', 'none'
cfg.data.normalize_naip = 'none'

# DEM gradient normalization. Options: 'instance' , 'none', '0_to_1', 'unit_gaussian', '0_to_255'
cfg.data.normalize_dem_ddxy = 'unit_gaussian'

# normalization for DEM pre-computed gradients. Options: 'none', 'instance', 'unit_gaussian', '0_to_1', '0_to_255'
cfg.data.normalize_dem_dxy_pre = '0_to_255'

cfg.data.mode = 'train'
cfg.data.cutout_size = (400, 400)

# Training details
cfg.train = edict()

cfg.train.batch_size = 14
cfg.train.learning_rate = 5e-4  # initial learning rate
cfg.train.l2_reg = 1e-6
cfg.train.lr_decay = 0.9
cfg.train.lr_decay_every = 10
cfg.train.shuffle = True
cfg.train.num_epochs = 100
cfg.train.num_workers = 4

cfg.train.out_dir = './outputs/dem_derovative1'