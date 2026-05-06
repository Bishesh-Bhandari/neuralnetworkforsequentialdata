import os                    # File path operations and also for crossplatform access
import re                    #  for text cleaning
import time                 
import math                  
import random              
import urllib.request        # Download dataset without extra dependencies
import json                  
from collections import Counter  # Count word frequencies for vocabulary

import yfinance as yf


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

STOCKS = ["AAPL", "TSLA"]          # two stocks
START_DATE = "2018-01-01"
END_DATE   = "2023-12-31"
FEATURE_COL = "Close"              #   we took closing price only as mention in assignment 


SEQ_LEN = 60                       # Use past 60 trading days to predict next day
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8


INPUT_SIZE = 1                  # only close price
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# Training
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
CLIP_GRAD_NORM = 5.0

print("Hyperparameters loaded")

# now we need to dowload the data n which we need to work on
def download_data(stocks,start_date,end_date):
    stock_data = {}

    for stock in stocks:
        print(f"downloading {stock} data....")
        df = yf.download(stock,start =start_date ,end =end_date,progress =False)

        df = df[[FEATURE_COL]].dropna()
        
        df.columns = ["Close"]

        stock_data[stock] = df
        print(f"{stock}: {len(df)} rows")

    return stock_data

stock_data = download_data(STOCKS, START_DATE, END_DATE)

total_steps = sum(len(df) for df in stock_data.values())
print(f"\nTotal time steps across stocks: {total_steps}")

#visualization of the raw data using matplotlib

plt.figure(figsize=(12, 5))

for ticker, df in stock_data.items():
    plt.plot(df.index, df["Close"], label=ticker)

plt.title("Historical Closing Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()