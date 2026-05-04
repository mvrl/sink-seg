# Automatic Segmentation of Sinkholes Using a Convolutional Neural Network

[M. Usman Rafique](http://urafique.com), [Nathan Jacobs](https://jacobsn.github.io/), [Junfeng Zhu](https://ees.as.uky.edu/users/jzh226)

## [Paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021EA002195) &emsp; [PDF](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021EA002195) &emsp; [Dataset](https://doi.org/10.5281/zenodo.5789436)

This repository contains PyTorch code for the following paper:

M. Usman Rafique, Junfeng Zhu, Nathan Jacobs "Automatic Segmentation of Sinkholes Using a Convolutional Neural Network", AGU Earth and Space Science Journal, 2022.

## Repository Structure

```
sink_seg/           # Python package (model, data, config)
  __init__.py
  config.py         # all configurable settings
  model.py          # UNet model definition
  data.py           # dataset class and dataloaders
train.py            # entry point: train a model
evaluate.py         # entry point: evaluate a trained model
visualize.py        # entry point: generate qualitative results
environment.yml     # conda environment
requirements.txt    # pip requirements
pyproject.toml      # packaging metadata
```

## Setting Up

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate sink-seg
pip install -e .
```

### Option B — pip only

```bash
pip install -r requirements.txt
pip install -e .
```

> **PyTorch note:** The commands above install the CPU-only build of PyTorch.  
> For GPU support, follow the instructions at <https://pytorch.org/get-started/locally/> to install a CUDA-enabled wheel before running the commands above.

### Dataset

Our dataset has been publicly released at Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5789436.svg)](https://doi.org/10.5281/zenodo.5789436)

Please download all the files into a directory named `data` inside this repository. The data is released under the Creative Commons Attribution 4.0 license.

## Configuration

All settings (input modality, normalization, batch size, learning rate, output directory, …) live in `sink_seg/config.py`. Edit that file before running any of the scripts below.

## Training

```bash
python train.py
```

The trained model checkpoint and training logs will be saved to the directory specified by `cfg.train.out_dir` in `sink_seg/config.py`.

## Evaluation

```bash
python evaluate.py
```

This performs a threshold sweep on the validation set, saves the best threshold to `best_threshold.txt`, and writes full metrics to `cfg.train.out_dir`.

## Visualization

```bash
python visualize.py
```

Saves 100 cutout-level qualitative results (DEM, soft prediction, binary prediction, ground-truth) to `<out_dir>/cutout_results/`.

## Acknowledgement

We thank Nicole Wong and Aram Ansary Ogholbake for testing the code.


## Citation
If you find this paper or code helpful, please cite our paper:
<pre>
@article{rafique2022automatic,
  title={Automatic Segmentation of Sinkholes using a Convolutional Neural Network},
  author={Rafique, M. Usman and Zhu, Junfeng and Jacobs, Nathan},
  journal={AGU Earth and Space Science Journal},
  year={2022},
  publisher={Wiley Online Library}
}
</pre>
