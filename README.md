# Project Information

This is an extension of the following papers:

Do ImageNet Classifiers Generalize to ImageNet?: https://arxiv.org/pdf/1902.10811.pdf

Do CIFAR-10 Classifiers Generalize to CIFAR-10?: https://arxiv.org/pdf/1806.00451.pdf

This project was run on Windows 10 Version 21H2 with WSL running Ubunutu 22.04

I have a NVIDIA 2070 Super with GeForce Game Ready Driver v526.47 


# Install Cuda

Install CUDA 11.6: https://developer.nvidia.com/cuda-11-6-0-download-archive

Verify installation:
```
$ /usr/local/cuda/bin/nvcc --version
> nvcc: NVIDIA (R) Cuda compiler driver
> Copyright (c) 2005-2022 NVIDIA Corporation
> Built on Tue_Mar__8_18:18:20_PST_2022
> Cuda compilation tools, release 11.6, V11.6.124
> Build cuda_11.6.r11.6/compiler.31057947_0
```


# Create venv and install required packages:

Python 3.7 and Python3.7-venv are required

Create venv, and install correct torch version and packages:
```
$ python3.7 -m venv {venv_name}
$ source {venv_name}/bin/activate
$ python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
$ python3 -m pip install -r requirements.txt
```

# Download CIFAR10.1
Clone the following repo in the same directory as this repo:

https://github.com/modestyachts/CIFAR-10.1

# Install pytorch_image_classification

Clone the following repo in the same directory as this repo:

https://github.com/hysts/pytorch_image_classification

I disabled NVIDA apex:

Change line 96 in pytorch_image_classification/pytorch_image_classificaion/config/defaults.py to ```config.train.use_apex = False```

The current directory should look like this:
```
masters_project/
    CIFAR-10.1/
    configs_files/
    {venv_name}/
    pytorch_image_classification/
    README.txt
    histogram.py
    model_test.py
    requirements.txt
```

# Download pretrained models

Models are too large to be saved on github, are available here: [Google Drive](https://drive.google.com/drive/folders/1vngHSSu4z-b7o7EqiCqK6Tr8sMesY0Nz?usp=sharing)

Place .pth files in /trained_models

# Train new models

Setup config file example:

To run resnext_29_4x64: 
configs/cifar/resnext.yaml
```
resnext:
    depth: 29
    initial_channels: 64
    cardinality: 8
    base_channels: 64
```
Other important settings:

```Train.batch_size``` Lower if running out of memory

```Train.output_dir``` Change between each new model
```
$ cd pytorch_image_classification
$ python3 train.py --config configs/cifar/{model_name}.yaml
```

# Evaluate All Trained Models on CIFAR10 and CIFAR10.1

If you trained new models, add the .pth file to ```trained_models```

```
$ python3 model_test.py {model_name}.pth
```

This should output accuracies on both test sets, and save the model outputs to ```/model_outputs```

# Generate Histograms

If you train new models, add the name of the model to ```model_list.txt.``` Ensure that ```/model_outputs``` contains ```{model_name}_cifar10.npz``` and ```{model_name}_cifar10-1.npz``` files.

```
$ python3 histogram.py
```

Outputs should be in ```/images```
