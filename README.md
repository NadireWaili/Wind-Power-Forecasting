# Wind Power Forecasting Using LSTM

A deep learning project implementing a Long Short-Term Memory (LSTM) recurrent neural network to forecast wind turbine power output (`ActivePower`) from meteorological and operational time-series data.

---

## 📌 Project Overview

Wind energy production is intermittent and weather-dependent. Accurate short-term wind power forecasting is vital for:
* Grid stability and energy dispatch planning.
* Balancing supply and demand in renewable energy integration.
* Minimizing economic penalties associated with forecast deviations.

This project implements a multi-feature, rolling-window LSTM pipeline to predict active power generation while strictly avoiding temporal data leakage.

---

## 📊 Dataset

The model processes wind turbine time-series telemetry data (e.g. `tatapower.csv`):
* **Sampling Resolution:** 10-minute intervals.
* **Key Features:**
  * `ActivePower(KW)`: Wind turbine active power generation (target variable).
  * `WindSpeed(m/sec)`: Anemometer wind speed measurement.
  * `NacelleDirection`: Turbine yaw / nacelle orientation (degrees).
  * `Temperature(c)`: Ambient temperature.
  * Cyclical Time Embeddings: Sine/Cosine encoding of time-of-day.

> **Note:** If `tatapower.csv` is not present, the script automatically generates a realistic synthetic wind dataset based on Weibull distribution and aerodynamic power curves for demonstration and testing.

---

## 🧠 Methodology & Architecture

1. **Feature Engineering & Cleaning:**
   * Dropped non-predictive identifiers and null values.
   * Encoded cyclic daily seasonality using $\sin(2\pi \cdot t / 24)$ and $\cos(2\pi \cdot t / 24)$ transforms.
2. **Data Leakage Prevention:**
   * Train/Test split chronologically (70% train, 30% test).
   * Feature scalers (`MinMaxScaler`) are fitted **strictly on the training partition**.
   * Target variable scaled independently for accurate inverse transformation.
3. **Supervised Sequence Windowing:**
   * Formatted input tensors into $[samples, lookback\_steps, features]$ using a sliding window of historical timesteps ($t-L, \dots, t-1 \to y_t$).
4. **Model Architecture:**
   * LSTM layer with 32 hidden units.
   * Dropout layer (0.1) for regularization.
   * Fully connected Dense output layer.
   * Optimized with Adam optimizer and Mean Squared Error (MSE) loss.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NadireWaili/Wind-Power-Forecasting.git
   cd Wind-Power-Forecasting
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
   ```

---

## 🚀 Usage

Run the end-to-end forecasting pipeline:
```bash
python3 wind_power_lstm_corrected.py
```

The script will:
1. Load or synthesize the dataset.
2. Build time sequences and train the LSTM network.
3. Evaluate test performance (Test RMSE & MAE in KW).
4. Save a prediction comparison figure to `predictions_plot.png`.

---

## 📈 Evaluation Metrics

Performance is quantified on unseen test intervals using:
* **Root Mean Squared Error (RMSE):** Measures standard deviation of forecast residuals.
* **Mean Absolute Error (MAE):** Average magnitude of absolute errors in kilowatts (KW).
