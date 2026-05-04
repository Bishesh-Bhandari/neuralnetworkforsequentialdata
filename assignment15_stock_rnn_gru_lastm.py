import os                    # File path operations and also for crossplatform access
import re                    #  for text cleaning
import time                 
import math                  
import random              
import urllib.request        # Download dataset without extra dependencies
import json                  
from collections import Counter  # Count word frequencies for vocabulary

import numpy as np           
import torch                
import torch.nn as nn       
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
import matplotlib
import matplotlib.pyplot as plt   # Plotting library for all charts
import matplotlib.gridspec as gridspec  # Advanced subplot layouts
matplotlib.rcParams['figure.dpi'] = 120   

print("All libraries imported successfully")
print(f"PyTorch version : {torch.__version__}")
print(f"NumPy version   : {np.__version__}")

# now we will fix the random seed for the reproducibility________
SEED = 42

random.seed(SEED)           
np.random.seed(SEED)       
torch.cuda.manual_seed_all(SEED)
print(f"All random seeds fixed to: {SEED}")

def get_device():

    
   # Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f" Using CUDA GPU: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" Using Apple MPS (Metal Performance Shaders) — M-series chip detected")
    else:
        device = torch.device("cpu")
        print(" Using CPU (no GPU acceleration available)")
    return device

DEVICE = get_device()
print(f"   Device object: {DEVICE}")

# we need to set the hyperparameters which are shared across rnn/lstm/gru

