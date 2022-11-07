# Project Information

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

```
$ python3.7 -m venv venv_name
$ source venv_name/bin/activate
$ python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
$ python3 -m pip install -r requirements.txt
```

