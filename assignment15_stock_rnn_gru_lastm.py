import os                    # File path operations and also for crossplatform access
import re                    #  for text cleaning
import time                 
import math                  
import random              
import urllib.request        # Download dataset without extra dependencies
import json                  
from collections import Counter  # Count word frequencies for vocabulary

import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np  
import pandas as pd         
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
torch.manual_seed(SEED)   
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

# ── Scale Data and Create Train/Test Split ────────────────────────────────
processed_data = {}

for ticker, df in stock_data.items():
    values = df[["Close"]].values.astype(np.float32)

    train_size = int(len(values) * TRAIN_SPLIT)

    train_values = values[:train_size]
    test_values  = values[train_size - SEQ_LEN:]  # include previous SEQ_LEN days for test windows

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values)
    test_scaled  = scaler.transform(test_values)

    processed_data[ticker] = {
        "df": df,
        "scaler": scaler,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "train_size": train_size
    }

    print(f"{ticker}: train={len(train_scaled)}, test={len(test_scaled)}")


#Sliding Window for Stock Prices

class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]       # shape: (seq_len, 1)
        y = self.data[idx + self.seq_len]             # shape: (1,)
        return x, y


# Create datasets and dataloaders for each stock
loaders = {}

for ticker, item in processed_data.items():
    train_dataset = StockDataset(item["train_scaled"], SEQ_LEN)
    test_dataset  = StockDataset(item["test_scaled"], SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    loaders[ticker] = {
        "train_loader": train_loader,
        "test_loader": test_loader
    }

    print(f"{ticker}: train sequences={len(train_dataset)}, test sequences={len(test_dataset)}")
    print("Dataset creation completed successfully")

    #Shared Architecture for RNN / LSTM / GRU
   
class RecurrentStockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, rnn_type):
        super().__init__()

        self.rnn_type = rnn_type.upper()

        if self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, hidden = self.rnn(x)

        # Use the output from the last time step
        last_out = out[:, -1, :]

        last_out = self.dropout(last_out)
        prediction = self.fc(last_out)

        return prediction
    
def create_model(rnn_type):
    torch.manual_seed(SEED)
    model = RecurrentStockPredictor(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        rnn_type=rnn_type
    ).to(DEVICE)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for model_name in ["RNN", "LSTM", "GRU"]:
    model = create_model(model_name)
    print(f"{model_name} parameters: {count_parameters(model):,}")


