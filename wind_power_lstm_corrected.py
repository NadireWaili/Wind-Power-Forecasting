"""
Corrected Wind Power Prediction using LSTM
This script resolves all data leakage, indexing, deprecation, and modeling bugs found in P1.ipynb.
Supports TensorFlow / Keras if installed, or PyTorch if installed.
"""

import os
from math import sqrt
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------------------------------------------------
# 1. Dataset Loading (with synthetic data generator fallback)
# -----------------------------------------------------------------------------
DATA_PATH = "tatapower.csv"

if os.path.exists(DATA_PATH):
    print(f"Loading real dataset from '{DATA_PATH}'...")
    data = pd.read_csv(DATA_PATH)
else:
    print(f"Notice: '{DATA_PATH}' not found in directory. Generating synthetic wind turbine dataset for demonstration...")
    np.random.seed(42)
    n_samples = 1200
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="10min")
    # Simulate realistic wind speed and cubic power relationship
    wind_speed = np.clip(np.random.weibull(2.0, n_samples) * 6.5, 0, 25)
    # Power curve approximation (cut-in ~ 3 m/s, rated ~ 12 m/s, cut-out ~ 25 m/s)
    rated_power = 1500.0
    active_power = np.where(
        wind_speed < 3.0, 0.0,
        np.where(
            wind_speed < 12.0,
            rated_power * ((wind_speed - 3.0) / 9.0) ** 3,
            np.where(wind_speed <= 25.0, rated_power, 0.0)
        )
    )
    # Add measurement noise
    active_power += np.random.normal(0, 25, n_samples)
    active_power = np.clip(active_power, 0, rated_power)

    data = pd.DataFrame({
        "Date/Time": dates,
        "ActivePower(KW)": active_power,
        "WindSpeed(m/sec)": wind_speed,
        "NacelleDirection": np.random.uniform(0, 360, n_samples),
        "Temperature(c)": 20 + 5 * np.sin(np.linspace(0, 10 * np.pi, n_samples)) + np.random.normal(0, 1, n_samples)
    })

# -----------------------------------------------------------------------------
# 2. Data Preprocessing & Feature Selection
# -----------------------------------------------------------------------------
data["Date/Time"] = pd.to_datetime(data["Date/Time"])
data = data.dropna()

# Extract cyclical time features (sin/cos encoding) instead of linear integers
hour = data["Date/Time"].dt.hour + data["Date/Time"].dt.minute / 60.0
data["Hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
data["Hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

# Define input features and target variable explicitly
FEATURE_COLS = ["ActivePower(KW)", "WindSpeed(m/sec)", "Hour_sin", "Hour_cos"]
TARGET_COL = "ActivePower(KW)"

feature_data = data[FEATURE_COLS].values
target_data = data[[TARGET_COL]].values

# -----------------------------------------------------------------------------
# 3. Train-Test Split & Scaling (Fit on Train ONLY to avoid Data Leakage)
# -----------------------------------------------------------------------------
train_size = int(len(data) * 0.7)

train_features, test_features = feature_data[:train_size], feature_data[train_size:]
train_target, test_target = target_data[:train_size], target_data[train_size:]

scaler_X = MinMaxScaler(feature_range=(-1, 1))
train_features_scaled = scaler_X.fit_transform(train_features)
test_features_scaled = scaler_X.transform(test_features)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
train_target_scaled = scaler_y.fit_transform(train_target)
test_target_scaled = scaler_y.transform(test_target)

# -----------------------------------------------------------------------------
# 4. Supervised Sequence Windowing (Past L steps -> Future step t)
# -----------------------------------------------------------------------------
def create_sequences(X_data, y_data, sequence_length=6):
    """Creates [samples, time_steps, features] windowed arrays."""
    X, y = [], []
    for i in range(sequence_length, len(X_data)):
        X.append(X_data[i - sequence_length:i, :])
        y.append(y_data[i, 0])
    return np.array(X), np.array(y)

LOOKBACK = 6  # 6 past steps (e.g. 1 hour lookback for 10-min data)
X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, sequence_length=LOOKBACK)
X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, sequence_length=LOOKBACK)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

# -----------------------------------------------------------------------------
# 5. Model Training (TensorFlow / PyTorch / Fallback)
# -----------------------------------------------------------------------------
predictions_scaled = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    print("\n--- Training LSTM with TensorFlow / Keras ---")
    model = Sequential([
        LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=1)
    predictions_scaled = model.predict(X_test)

except ImportError:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        print("\n--- TensorFlow not found. Training LSTM with PyTorch ---")
        class PyTorchLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim=1):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PyTorchLSTM(X_train.shape[2], 32).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(-1))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)

        model.train()
        for epoch in range(15):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/15 - Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test).to(device))
            predictions_scaled = preds.cpu().numpy()

    except ImportError:
        print("\nNeither TensorFlow nor PyTorch is installed.")
        print("To run the deep learning training, install either:")
        print("  pip install tensorflow")
        print("  or")
        print("  pip install torch")

# -----------------------------------------------------------------------------
# 6. Evaluation & Inverse Scaling (Target ONLY)
# -----------------------------------------------------------------------------
if predictions_scaled is not None:
    # Invert scaling strictly on target variable
    predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
    actuals = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    print(f"\nModel Performance on Test Set:")
    print(f"Test RMSE: {rmse:.3f} KW")
    print(f"Test MAE:  {mae:.3f} KW")

    # -----------------------------------------------------------------------------
    # 7. Visualization
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(actuals[:100], label="Actual ActivePower (KW)", color="steelblue")
    plt.plot(predictions[:100], label="Predicted ActivePower (KW)", color="darkorange", linestyle="--")
    plt.xlabel("Sample Index (10-minute intervals)")
    plt.ylabel("Active Power (KW)")
    plt.title("Wind Power Forecasting: Actual vs Predicted (LSTM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions_plot.png")
    print("Saved predictions plot to 'predictions_plot.png'.")
