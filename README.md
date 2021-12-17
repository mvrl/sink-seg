# Sinkhole Segmentation

This repository contains PyTorch code for the following paper:

M. Usman Rafique, Junfeng Zhu, Nathan Jacobs "Automatic Segmentation of Sinkholes Using a Convolutional Neural Network", under review.

## Setting Up

### Installation

We recommend using a virtual environment with anaconda. If you don't already have anaconda installed, please visit this page to install anaconda: [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html)

We provide a virtual environment  file `environment.yml` in this repository that can be used to make a suitable virtual environment. In terminal, go to the path of this repo and use this command:

`conda env create -f environment.yml`

Once you set up the virtual environment (named  `sinkhole`), you can activate it by:

`conda activate sinkhole`

### Data

Copy the all the files in a directory named `data` in this repository. Please contact junfeng.zhu@uky.edu to get access to the data while we work on public release of the data.

## Training

To train a model, first set up the required settings in the `config.py` file. The default settings use two-channel DEM derivatives as input and a Unit Gaussian normalization. Other inputs and normalization options are available. Additionally, data and optimization settings can also be set in the config file.

To start training, simply use the command:

`python3 train.py`

At the end of the training, the trained model checkpoint and training logs will be saved in the `cfg.train.out_dir` specified in the `config.py` file.

## Evaluation

To evaluate a trained model, set the same `cfg.train.out_dir` in `config.py` that was used to train the model and run the command:

`python3 evaluate.py`

Metrics will be displayed in the terminal and saved to the `cfg.train.out_dir` directory. This script performs a sweep of different thresholds on the val set and saves the best threshold in a file `best_threshold.txt` as well.

## Visualization

To generate qualitative visual results, run the command 

`python3 visualize.py`

As before, it is required that `cfg.train.out_dir` in `config.py` specifies a directory that contains a trained model checkpoint.

## Acknowledgement

We thank Nicole Wong and Aram Ansary Ogholbake for testing the code.
