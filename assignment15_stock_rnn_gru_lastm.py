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


 #Loss Function and Optimizer
criterion = nn.MSELoss()

def make_optimizer(model):
    return optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Loss function: MSELoss")
print("Optimizer: Adam")

# Training and Evaluation Functions 
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)

        loss = criterion(preds, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    epoch_time = time.time() - start_time

    return avg_loss, epoch_time


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


#Master Training Function
def train_model_for_stock(ticker, model_name):
    train_loader = loaders[ticker]["train_loader"]
    test_loader  = loaders[ticker]["test_loader"]

    model = create_model(model_name)
    optimizer = make_optimizer(model)

    history = {
        "train_loss": [],
        "test_loss": [],
        "epoch_times": []
    }

    print(f"\nTraining {model_name} for {ticker}")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, epoch_time = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss = evaluate_loss(model, test_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["epoch_times"].append(epoch_time)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test Loss: {test_loss:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

    return model, history

#train each model for each stock
results = {}

for ticker in STOCKS:
    results[ticker] = {}

    for model_name in ["RNN", "LSTM", "GRU"]:
        model, history = train_model_for_stock(ticker, model_name)

        results[ticker][model_name] = {
            "model": model,
            "history": history
        }
# Prediction Function 
def get_predictions(model, loader, device):
    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            output = model(x_batch).cpu().numpy()
            target = y_batch.numpy()

            preds.append(output)
            actuals.append(target)

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    return preds, actuals

#  Compute RMSE, MAE, and MAPE in Original Price Scale
summary_rows = {}

for ticker in STOCKS:
    summary_rows[ticker] = {}

    scaler = processed_data[ticker]["scaler"]
    test_loader = loaders[ticker]["test_loader"]

    for model_name in ["RNN", "LSTM", "GRU"]:
        model = results[ticker][model_name]["model"]

        preds_scaled, actuals_scaled = get_predictions(model, test_loader, DEVICE)

        preds = scaler.inverse_transform(preds_scaled)
        actuals = scaler.inverse_transform(actuals_scaled)

        rmse = math.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

        summary_rows[ticker][model_name] = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "preds": preds,
            "actuals": actuals
        }

        print(f"{ticker} {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")


# ── Final Results Table ───────────────────────────────────────────────────
table_rows = []

for ticker in STOCKS:
    for model_name in ["RNN", "LSTM", "GRU"]:
        metrics = summary_rows[ticker][model_name]
        avg_time = np.mean(results[ticker][model_name]["history"]["epoch_times"])

        table_rows.append({
            "Stock": ticker,
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE (%)": metrics["MAPE"],
            "Avg Epoch Time (s)": avg_time
        })

summary_df = pd.DataFrame(table_rows)
summary_df



# ── Plot Training and Test Loss Curves ────────────────────────────────────
for ticker in STOCKS:
    plt.figure(figsize=(12, 5))

    for model_name in ["RNN", "LSTM", "GRU"]:
        h = results[ticker][model_name]["history"]
        plt.plot(h["test_loss"], marker="o", label=f"{model_name} Test Loss")

    plt.title(f"{ticker}: Validation/Test Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


    # ── Plot Actual vs Predicted Prices ───────────────────────────────────────
for ticker in STOCKS:
    actual = summary_rows[ticker]["RNN"]["actuals"]

    plt.figure(figsize=(14, 6))
    plt.plot(actual, label="Actual", linewidth=2)

    for model_name in ["RNN", "LSTM", "GRU"]:
        preds = summary_rows[ticker][model_name]["preds"]
        plt.plot(preds, label=f"{model_name} Predicted", alpha=0.8)

    plt.title(f"{ticker}: Actual vs Predicted Closing Price")
    plt.xlabel("Test Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()



    # ── Bar Chart: RMSE Comparison ────────────────────────────────────────────
for ticker in STOCKS:
    model_names = ["RNN", "LSTM", "GRU"]
    rmse_values = [summary_rows[ticker][m]["RMSE"] for m in model_names]

    plt.figure(figsize=(7, 5))
    plt.bar(model_names, rmse_values)
    plt.title(f"{ticker}: RMSE Comparison")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.grid(axis="y")
    plt.show()



    # ── Find Best Model for Each Stock ────────────────────────────────────────
for ticker in STOCKS:
    best_model = min(
        ["RNN", "LSTM", "GRU"],
        key=lambda m: summary_rows[ticker][m]["RMSE"]
    )

    print(f"{ticker}: Best model based on RMSE = {best_model}")
    print(f"RMSE: {summary_rows[ticker][best_model]['RMSE']:.4f}")
    print(f"MAE : {summary_rows[ticker][best_model]['MAE']:.4f}")
    print(f"MAPE: {summary_rows[ticker][best_model]['MAPE']:.2f}%")
   