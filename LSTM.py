from datetime import datetime

import kagglehub
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras.src.layers import Bidirectional, LSTM, BatchNormalization, Dropout, Dense
from keras.src.optimizers import Adam
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2



def main():
    print("Main method executed.")
    # Add your main logic here

    # Download latest version
    path = kagglehub.dataset_download("atomicd/retail-store-inventory-and-demand-forecasting")
    print("Path to dataset files:", path)

    # List all files in the downloaded folder
    print(os.listdir(path))

    csv_file = os.path.join(path, "sales_data.csv")
    df = pd.read_csv(csv_file)

    df_numerical = df.select_dtypes(include=["int64", "float64"])
    # Check for correlation between features
    plt.figure(figsize=(16, 12))
    sns.heatmap(df_numerical.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    df_groceries = df[(df['Category'] == 'Groceries')]
    #df_groceries = df[(df['Category'] == 'Groceries') & (df['Store ID'] == 'S001') ]
    df_groceries = df[(df['Category'] == 'Groceries') & (~df['Epidemic']) & (~df['Promotion'])]
    df_groceries.head()
    df_groceries.info()
    df_groceries.describe()
    df_groceries.isnull().sum()

    # Initial Data Visualization
    # Plot Prices of Groceries over time
    plt.figure(figsize=(12, 6))
    plt.plot(df_groceries['Date'], df_groceries['Price'], label='Price', color='blue')
    plt.plot(df_groceries['Date'], df_groceries['Competitor Pricing'], label='Competitor Pricing', color='green')
    plt.title('Price and Competitor Pricing  of Groceries Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price/Competitor Pricing')
    plt.legend()
    plt.show()


    # Convert date and sort
    df_groceries['Date'] = pd.to_datetime(df_groceries['Date'])
    df_groceries = df_groceries.sort_values('Date').reset_index(drop=True)

    # Advanced feature engineering
    # Time-based features
    df_groceries['day_of_week'] = df_groceries['Date'].dt.dayofweek
    df_groceries['month'] = df_groceries['Date'].dt.month
    df_groceries['quarter'] = df_groceries['Date'].dt.quarter
    df_groceries['is_weekend'] = (df_groceries['Date'].dt.dayofweek >= 5).astype(int)
    df_groceries['day_of_month'] = df_groceries['Date'].dt.day
    df_groceries['week_of_year'] = df_groceries['Date'].dt.isocalendar().week

    # Lag features
    for lag in [1, 3, 7, 14]:
        df_groceries[f'price_lag_{lag}'] = df_groceries['Price'].shift(lag)
        df_groceries[f'demand_lag_{lag}'] = df_groceries['Demand'].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df_groceries[f'price_ma_{window}'] = df_groceries['Price'].rolling(window=window, center=True).mean()
        df_groceries[f'price_std_{window}'] = df_groceries['Price'].rolling(window=window, center=True).std()
        df_groceries[f'demand_ma_{window}'] = df_groceries['Demand'].rolling(window=window, center=True).mean()

    # Price volatility
    df_groceries['price_volatility_7'] = df_groceries['Price'].rolling(window=7).std()
    df_groceries['price_change'] = df_groceries['Price'].pct_change()
    df_groceries['demand_change'] = df_groceries['Demand'].pct_change()

    # Interaction features
    df_groceries['price_demand_ratio'] = df_groceries['Price'] / (df['Demand'] + 1e-8)
    df_groceries['price_inventory_interaction'] = df_groceries['Price'] * df['Inventory Level']

    df_groceries = df_groceries.fillna(method='ffill').fillna(method='bfill')

    print(f"Data shape after preprocessing: {df_groceries.shape}")

    #Prepare the data
    #Select numerical features
    feature_cols = [col for col in df.columns if col not in ['Date', 'Category', 'Store ID', 'Product ID', 'Region', 'Weather Condition', 'Seasonality']]
    df_features = df_groceries[feature_cols].copy()

    # Get target column index (Price)
    target_col_idx = df_features.columns.get_loc('Price')

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features.values)

    # Calculate split points
    total_len = len(scaled_data)
    train_end = int(total_len * (1 - 0.15 - 0.15))
    val_end = int(total_len * (1 - 0.15))

    # Split data
    train_data = scaled_data[:train_end]
    val_data = scaled_data[:val_end]  # Include training data for validation sequences
    test_data = scaled_data[:]  # Include all data for test sequences

    # Create sequences for training
    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i - 60:i, :])  # All features
        y_train.append(train_data[i, target_col_idx])  # Target (Price)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Create sequences for validation
    X_val, y_val = [], []
    for i in range(60, len(val_data)):
        X_val.append(val_data[i - 60:i, :])  # All features
        y_val.append(val_data[i, target_col_idx])  # Target (Price)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Create sequences for testing
    X_test, y_test = [], []
    for i in range(60, len(test_data)):
        X_test.append(test_data[i - 60:i, :])
        y_test.append(test_data[i, target_col_idx])  # Target (Price)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Only use validation portion for validation
    val_start_idx = train_end - 60
    X_val = X_val[val_start_idx:]
    y_val = y_val[val_start_idx:]

    # Only use test portion for testing
    test_start_idx = val_end - 60
    X_test = X_test[test_start_idx:]
    y_test = y_test[test_start_idx:]

    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    print(f"Test sequences: {X_test.shape}")

    feature_names = df_groceries.columns

    # 3D Array for tensor flow
    # X_train shape: (samples, time_steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    #keras used for building and training neural networks
    model = keras.models.Sequential()
    # Creating 5 layer model
    # LTSM Layers - Understand patterns over time
    # Dense Layers - Learn more complex relationships
    # Dropout Layers - Prevent overfitting
    # Final Output Layer - Make the final prediction

    # # First Layer like a brain to help the model understand patterns. Learns pattern from past data over time
    # # 64 Number of memory cells, return_sequence = after done get me the full list of ideas for the next layer, input_shape = what kind of data to expect
    model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    # # Second Layer Layer distills the important patterns from the first layer an dprepares to make a final prediction
    model.add(keras.layers.LSTM(56, return_sequences=False))
    # # Third Dense Layer turns complex patterns into a decision
    # # 128 regular neural network with 128 neurons to make final decision
    # # Activation helps the model introduce non-linearity
    model.add(keras.layers.Dense(128, activation='relu'))
    # # Fourth Dropout layer - randomly drops out 50% of the neurons during training to help pevent overfitting.
    # # Helps keep model from being too sensitive to the training data and makes it perform better
    model.add(keras.layers.Dropout(0.5))
    # # Fifth - Final Dense layer - makes a final prediction
    # # Simple layer with 1 neuron that output the predicted value
    model.add(keras.layers.Dense(1))
    #
    model.summary()
    #
    # # Model Compilation - This compiles for the model to tell it how to learn
    # # Optimizer - Adam popular optimizer theat helps the model adjust its learning. Works well with time serries data
    # # loss - MAE measures how far the precitions are from the acutal value. Lower MAE is better.
    # # metrics - Another performance metric using RootMeanSquareError to check how accurate the pred
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mae', metrics=['RootMeanSquaredError'])

    # model = Sequential([
    #     # First Bidirectional LSTM layer
    #     Bidirectional(LSTM(64, return_sequences=True,
    #                        kernel_regularizer=l1_l2(0.01, 0.01)),
    #                   input_shape=(X_train.shape[1], X_train.shape[2])),
    #     BatchNormalization(),
    #     Dropout(0.2),
    #
    #     # Second LSTM layer
    #     LSTM(32, return_sequences=False,
    #          kernel_regularizer=l1_l2(0.01, 0.01)),
    #     BatchNormalization(),
    #     Dropout(0.3),
    #
    #     # Dense layers
    #     Dense(50, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
    #     BatchNormalization(),
    #     Dropout(0.3),
    #
    #     Dense(25, activation='relu'),
    #     Dropout(0.2),
    #
    #     # Output layer
    #     Dense(1)
    # ])
    #
    # optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    # model.compile(
    #     optimizer=optimizer,
    #     loss='huber',  # More robust to outliers
    #     metrics=['mae', 'mse']
    # )
    #
    # model.summary()

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # training
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=20, callbacks=callbacks, verbose=1)

    #prediction
    predictions = model.predict(X_test)

    # Create dummy array for inverse transform
    dummy_array = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy_array[:, -1] = predictions.flatten()  # Assuming Price is last column
    predictions_rescaled = scaler.inverse_transform(dummy_array)[:, -1]

    dummy_array[:, -1] = y_test.flatten()
    y_test_rescaled = scaler.inverse_transform(dummy_array)[:, -1]

    # Calculate metrics
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, predictions_rescaled)
    mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled)

    print(f"\nModel Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.4f}%")

    # Plotting the predictions vs actual values
    # Calculate split points for plotting
    total_len = len(df_groceries)
    train_end = int(total_len * (1 - 15 - 15))
    val_end = int(total_len * (1 - 15))

    # Create subsets for plotting
    df_train = df_groceries[:train_end]
    df_val = df_groceries[train_end:val_end]
    df_test = df_groceries[val_end:]

    # Align predictions with test data
    test_start_idx = len(df_groceries) - len(predictions_rescaled)
    df_test_aligned = df_groceries[test_start_idx:].copy()
    df_test_aligned['Predictions'] = predictions_rescaled

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Plot 1: Full time series with predictions
    axes[0, 0].plot(df_train['Date'], df_train['Price'], label='Training', color='blue', alpha=0.7)
    axes[0, 0].plot(df_val['Date'], df_val['Price'], label='Validation', color='orange', alpha=0.7)
    axes[0, 0].plot(df_test_aligned['Date'], df_test_aligned['Price'], label='Actual Test', color='green')
    axes[0, 0].plot(df_test_aligned['Date'], df_test_aligned['Predictions'], label='Predictions', color='red',
                    linestyle='--')
    axes[0, 0].set_title('Price Prediction - Full Timeline')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Training history
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Training History - Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    # Plot 3: Actual vs Predicted scatter
    axes[1, 0].scatter(y_test_rescaled, predictions_rescaled, alpha=0.6)
    axes[1, 0].plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 'r--', lw=2)
    axes[1, 0].set_title('Actual vs Predicted Values')
    axes[1, 0].set_xlabel('Actual Price')
    axes[1, 0].set_ylabel('Predicted Price')

    # Plot 4: Residuals
    residuals = y_test_rescaled - predictions_rescaled
    axes[1, 1].scatter(predictions_rescaled, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].set_xlabel('Predicted Price')
    axes[1, 1].set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
