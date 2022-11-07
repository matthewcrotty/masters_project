# Install Cuda

Install CUDA 11.6: https://developer.nvidia.com/cuda-11-6-0-download-archive
Verify installation:
/usr/local/cuda/bin/nvcc --version

# Create venv:

```
$ python3.7 -m venv venv_name
$ source venv_name/bin/activate
$ python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
$ python3 -m pip install -r requirements.txt
```

