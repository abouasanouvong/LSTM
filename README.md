# LSTM Time Series Forecasting for Retail Price Prediction

This project uses a Long Short-Term Memory (LSTM) neural network to forecast the price of a specific grocery product based on historical sales data. It includes data preprocessing, advanced feature engineering, model training, and evaluation.

## Dataset

The project utilizes the [Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting) dataset from Kaggle. The script specifically filters for:
- **Category**: Groceries
- **Store ID**: S001
- **Product ID**: P0005

## Features

The model is trained on a rich set of features engineered from the raw data, including:
- **Time-based features**: Day of the week, month, quarter, etc.
- **Lag features**: Price and demand from previous days (e.g., 1, 3, 7, 14 days ago).
- **Rolling statistics**: Moving averages and standard deviations for price and demand over different windows (e.g., 7, 14, 30 days).
- **Interaction features**: Ratios and products of different variables like price and inventory.

## Model Architecture

The forecasting model is a sequential LSTM network built with Keras, consisting of:
- An LSTM layer to capture temporal patterns.
- A second LSTM layer to distill important information.
- Dense layers for learning complex relationships.
- Dropout layers to prevent overfitting.
- A final Dense layer to output the price prediction.

The model is trained using the Adam optimizer and Mean Absolute Error (MAE) loss function. Callbacks like `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` are used to optimize the training process.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    Make sure you have Python installed. Then, install the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script:**
    Execute the main Python script to start the data download, preprocessing, training, and evaluation process.
    ```bash
    python LSTMV2.py
    ```

## Files

- `LSTMV2.py`: The main script containing all the logic for data handling, feature engineering, model building, and evaluation.
- `requirements.txt`: A list of all the Python packages required to run the project.
- `best_model.h5`: The saved weights of the best-performing model, generated during training.

