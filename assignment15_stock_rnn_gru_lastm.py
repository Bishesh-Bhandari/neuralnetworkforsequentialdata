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