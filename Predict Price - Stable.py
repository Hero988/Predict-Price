import pandas as pd
import datetime
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
import pytz
import pandas_ta as ta
from sklearn.model_selection import train_test_split

import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from collections import namedtuple
torch.distributions.Categorical

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Modified for N-step Learning
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'n_step_reward', 'n_step_state'))

# Your MetaTrader 5 login details
account_number = 1058146570  # Replace with your account number
password = 'ieasDILwkA8*'  # Replace with your password
server_name ='FTMO-Demo'

def mt5_login(account_number, password, server_name):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    # Use the server_name parameter in the login call
    authorized = mt5.login(account_number, password=password, server=server_name)
    if not authorized:
        print("login failed, error code =", mt5.last_error())
        mt5.shutdown()
        quit()
    else:
        print("Connected to MetaTrader 5")

def fetch_and_prepare_fx_data_mt5(symbol, timeframe_str, start_date, end_date):

    mt5_login(account_number, password, server_name)

    # Set time zone to UTC
    timezone = pytz.timezone("Europe/Berlin")

    # Convert string dates to datetime objects in UTC
    start_date = start_date.replace(tzinfo=timezone)
    end_date = end_date.replace(hour=23, minute=59, second=59, tzinfo=timezone)

    # Define the timeframe mapping correctly
    timeframes = {
        '1H': mt5.TIMEFRAME_H1,
        'DAILY': mt5.TIMEFRAME_D1,
        '12H': mt5.TIMEFRAME_H12,
        '2H': mt5.TIMEFRAME_H2,
        '3H': mt5.TIMEFRAME_H3,
        '4H': mt5.TIMEFRAME_H4,
        '6H': mt5.TIMEFRAME_H6,
        '8H': mt5.TIMEFRAME_H8,
        '1M': mt5.TIMEFRAME_M1,
        '10M': mt5.TIMEFRAME_M10,
        '12M': mt5.TIMEFRAME_M12,
        '15M': mt5.TIMEFRAME_M15,
        '2M': mt5.TIMEFRAME_M2,
        '20M': mt5.TIMEFRAME_M20,
        '3M': mt5.TIMEFRAME_M3,
        '30M': mt5.TIMEFRAME_M30,
        '4M': mt5.TIMEFRAME_M4,
        '5M': mt5.TIMEFRAME_M5,
        '6M': mt5.TIMEFRAME_M6,
        '1MN': mt5.TIMEFRAME_MN1,
        '1W': mt5.TIMEFRAME_W1
    }

    # Access the correct timeframe using the provided string key
    timeframe = timeframes.get(timeframe_str)
    if timeframe is None:
        print(f"Invalid timeframe: {timeframe_str}")
        mt5.shutdown()
        return None

    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    if rates is None:
        print("No rates retrieved, error code =", mt5.last_error())
        mt5.shutdown()
        return None
    
    # Convert the rates to a DataFrame
    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

    # Set the time as the index and convert to the proper format
    rates_frame.set_index('time', inplace=True)
    rates_frame.index = pd.to_datetime(rates_frame.index, format="%Y-%m-%d %H:%M:%S")

    if 'tick_volume' not in rates_frame.columns:
        print("tick_volume is not in the fetched data. Ensure it's included in the API call.")
    else:
        print("tick_volume is included in the data.")
    
    # Shutdown the MT5 connection
    mt5.shutdown()
    
    return rates_frame

# Get the current date
current_date = str(datetime.now().date())

def get_user_date_input(prompt):
    """
    Prompts the user for a date input and returns the date in 'YYYY-MM-DD' format.
    Continues to prompt until a valid date is entered.
    """
    date_format = '%Y-%m-%d'
    date_str = input(prompt)
    # Try to convert the user input into a datetime object to validate it
    while True:
        try:
            # If successful, return the string as it is valid
            pd.to_datetime(date_str, format=date_format)
            return date_str
        except ValueError:
            # If there's a ValueError, it means the format is incorrect. Prompt the user again.
            print("The date format is incorrect. Please enter the date in 'YYYY-MM-DD' format.")
            date_str = input(prompt)

strategy_start_date_all = "1971-01-04"
strategy_end_date_all = current_date

# Convert string dates to datetime objects (for main data)
start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

# Now call the function with the datetime objects
timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()  # or "1H", depending on what you need
# Prompt the user to enter the currency pair
Pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

eur_usd_data = fetch_and_prepare_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

def calculate_indicators(data, bollinger_length=12, bollinger_std_dev=1.5, sma_trend_length=50, window=9):
    # Calculate the 50-period simple moving average of the 'close' price
    data['SMA_50'] = ta.sma(data['close'], length=50)
    # Calculate the 200-period simple moving average of the 'close' price
    data['SMA_200'] = ta.sma(data['close'], length=200)
    
    # Calculate the 50-period exponential moving average of the 'close' price
    data['EMA_50'] = ta.ema(data['close'], length=50)
    # Calculate the 200-period exponential moving average of the 'close' price
    data['EMA_200'] = ta.ema(data['close'], length=200)

    # Calculate the 9-period exponential moving average for scalping strategies
    data['EMA_9'] = ta.ema(data['close'], length=9)
    # Calculate the 21-period exponential moving average for scalping strategies
    data['EMA_21'] = ta.ema(data['close'], length=21)
    
    # Generate original Bollinger Bands with a 20-period SMA and 2 standard deviations
    original_bollinger = ta.bbands(data['close'], length=20, std=2)
    # The 20-period simple moving average for the middle band
    data['SMA_20'] = ta.sma(data['close'], length=20)
    # Upper and lower bands from the original Bollinger Bands calculation
    data['Upper Band'] = original_bollinger['BBU_20_2.0']
    data['Lower Band'] = original_bollinger['BBL_20_2.0']

    # Generate updated Bollinger Bands for scalping with custom length and standard deviation
    updated_bollinger = ta.bbands(data['close'], length=bollinger_length, std=bollinger_std_dev)
    # Assign lower, middle, and upper bands for scalping
    data['Lower Band Scalping'], data['Middle Band Scalping'], data['Upper Band Scalping'] = updated_bollinger['BBL_'+str(bollinger_length)+'_'+str(bollinger_std_dev)], ta.sma(data['close'], length=bollinger_length), updated_bollinger['BBU_'+str(bollinger_length)+'_'+str(bollinger_std_dev)]
    
    # Calculate the MACD indicator and its signal line
    macd = ta.macd(data['close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['Signal_Line'] = macd['MACDs_12_26_9']
    
    # Calculate the Relative Strength Index (RSI) with the specified window length
    data[f'RSI_{window}'] = ta.rsi(data['close'], length=window).round(2)

    # Calculate a 5-period RSI for scalping strategies
    data[f'RSI_5 Scalping'] = ta.rsi(data['close'], length=5).round(2)

    # Calculate a simple moving average for trend analysis in scalping strategies
    data[f'SMA_{sma_trend_length}'] = ta.sma(data['close'], length=sma_trend_length)

    # Calculate the Stochastic Oscillator
    stoch = ta.stoch(data['high'], data['low'], data['close'])
    data['Stoch_%K'] = stoch['STOCHk_14_3_3']
    data['Stoch_%D'] = stoch['STOCHd_14_3_3']

    # Add column for Previous Close Price
    data['Previous Close'] = data['close'].shift(1)

    # Assuming 'data' is your DataFrame and it already includes 'close' and 'Previous Close' columns
    data['Actual Price Movement'] = np.where(data['close'] > data['Previous Close'], 1, 
                                            np.where(data['close'] < data['Previous Close'], -1, 0))

    # Convert 'Actual Price Movement' to integer
    data['Actual Price Movement'] = data['Actual Price Movement'].astype(int)

    # Add column for Previous Price Movement Price
    data['Previous Actual Price Movement'] = data['Actual Price Movement'].shift(1)

    # Return the data with added indicators
    return data

def create_diagrams(data, name_of_folder):

    # Ensure the Graphs directory exists
    graphs_dir = f'Graphs, {name_of_folder}'
    os.makedirs(graphs_dir, exist_ok=True)

    plt.ioff()  # Interactive mode off

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['close'], label='Close Price')
    plt.plot(data.index, data['SMA_50'], label='50-Day SMA', alpha=0.75)
    plt.plot(data.index, data['SMA_200'], label='200-Day SMA', alpha=0.75)
    plt.plot(data.index, data['EMA_50'], label='50-Day EMA', color='cyan', alpha=0.75)
    plt.plot(data.index, data['EMA_200'], label='200-Day EMA', color='magenta', alpha=0.75)

    # Bollinger Bands
    plt.plot(data.index, data['SMA_20'], label='20-Day SMA', color='orange', alpha=0.75)
    plt.plot(data.index, data['Upper Band'], label='Upper Bollinger Band', linestyle='--', color='grey')
    plt.plot(data.index, data['Lower Band'], label='Lower Bollinger Band', linestyle='--', color='grey')
    plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.1)

    plt.title('EUR/USD with SMAs, EMAs, and Bollinger Bands')
    plt.savefig(f'{graphs_dir}/price_smas_emas_bollinger_bands.png')
    plt.legend()
    plt.close()

    # Plotting RSI with thresholds
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data[f'RSI_9'], label='RSI', color='red', alpha=0.75)

    # Add dotted lines at RSI 70 and 30
    plt.axhline(70, color='green', linestyle='dotted', label='Overbought (70)')
    plt.axhline(30, color='blue', linestyle='dotted', label='Oversold (30)')

    plt.title('RSI with Overbought and Oversold Levels')
    plt.legend()
    plt.savefig(f'{graphs_dir}/rsi.png')
    plt.close()

    # Plotting the table of the last 20 rows in a new figure
    plt.figure(figsize=(14, 7))
    table_data = data.tail(20).reset_index()  # Reset index to turn the DateTime index into a column

    # Select only the desired columns for the table
    table_data = table_data[['time', 'close', 'Previous Close', 'Actual Price Movement', 'Previous Actual Price Movement', 'EMA_200',]]

    ax_table = plt.subplot(111, frame_on=False)  # no visible frame
    ax_table.xaxis.set_visible(False)  # hide the x axis
    ax_table.yaxis.set_visible(False)  # hide the y axis
    table = plt.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(f'Last 20 Rows of {Pair} Data')
    plt.savefig(f'{graphs_dir}/last_20_rows_table.png')
    plt.close()

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['MACD'], label='MACD', color='blue')
    plt.plot(data.index, data['Signal_Line'], label='Signal Line', color='red')
    plt.title('MACD and Signal Line')
    plt.legend()
    plt.savefig(f'{graphs_dir}/macd_signal_line.png')
    plt.close()

def create_inout_sequences(input_data, target_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = torch.tensor(input_data[i:i+tw], dtype=torch.float32)
        train_label = torch.tensor(target_data[i+tw], dtype=torch.float32)
        inout_seq.append((train_seq, train_label.unsqueeze(-1)))  # Ensure the label is correctly shaped
    return inout_seq

def preprocess_data(data, feature_columns, target_column, sequence_length):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Scale features
    data_scaled_features = scaler.fit_transform(data[feature_columns])
    
    # Ensure target column is correctly shaped and scaled
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled_target = target_scaler.fit_transform(data[[target_column]].values).flatten()  # Use flatten() for compatibility
    
    sequences = create_inout_sequences(data_scaled_features, data_scaled_target, sequence_length)
    
    # Convert sequences to a DataLoader
    sequence_tensors = [torch.tensor(s[0], dtype=torch.float32) for s in sequences]
    target_tensors = [torch.tensor(s[1], dtype=torch.float32) for s in sequences]

    dataset = TensorDataset(torch.stack(sequence_tensors), torch.stack(target_tensors))
    
    return DataLoader(dataset, batch_size=64, shuffle=True)
   
class LSTM_Actual_Price_Movement_Prediction(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, input_seq):
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size, dtype=torch.float32)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size, dtype=torch.float32)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.sigmoid(self.linear(lstm_out[:, -1, :]))  # Apply sigmoid
        return predictions
    

saved_to_file = False

# Apply technical indicators to the data using the 'calculate_indicators' function
eur_usd_data = calculate_indicators(eur_usd_data)

model_path_actual_price_movement = f"lstm_model_{Pair}_{timeframe_str}_actual_price_movement.pth"

is_training_actual_price_movement = input("Do you want to train the model for the actual price movement? (yes or no)").lower().strip() == 'yes'

if is_training_actual_price_movement:
    start_date_test_training = get_user_date_input("Please enter the start date for training data (YYYY-MM-DD): ")
    end_date_test_training = get_user_date_input("Please enter the end date for training data (YYYY-MM-DD): ")

    # Filter the EUR/USD data for the in-sample training period
    strategy_data_in_sample = eur_usd_data[(eur_usd_data.index >= start_date_test_training) & (eur_usd_data.index <= end_date_test_training)]

    # Select the features you want to use in the model
    feature_columns = ['Previous Actual Price Movement']  # Adjust as needed
    target_column = 'Actual Price Movement'  # Use string for single column

    # Data preprocessing
    sequence_length = 1  # The number of past days to consider for predicting the next day's close price

    # Calculate the split index
    split_idx = int(len(strategy_data_in_sample) * 0.8)

    # Split the data
    training_data = strategy_data_in_sample.iloc[:split_idx]
    validation_data = strategy_data_in_sample.iloc[split_idx:]

    # Preprocess both training and validation data, adjust function call to match new definition
    train_loader = preprocess_data(training_data, feature_columns, target_column, sequence_length)
    val_loader = preprocess_data(validation_data, feature_columns, target_column, sequence_length)
    
    # Model setup
    model = LSTM_Actual_Price_Movement_Prediction(input_size=len(feature_columns), hidden_layer_size=100, output_size=1)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Early stopping setup
    patience = 50
    patience_counter = 0
    best_loss = float('inf')

    epochs = int(input("Please input the number of epochs you would like to run: "))

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            y_pred = torch.squeeze(y_pred)
            labels = torch.squeeze(labels)
            # Example adjustment before loss calculation
            labels_adjusted = (labels + 1) / 2  # Converts -1, 1 to 0, 1
            loss = loss_function(y_pred, labels_adjusted)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = 0
        model.eval()
        actuals, predictions = [], []
        with torch.no_grad():
            for seq, labels in val_loader:
                y_pred = model(seq)
                y_pred = torch.squeeze(y_pred)
                labels = torch.squeeze(labels)
                # Example adjustment before loss calculation
                labels_adjusted = (labels + 1) / 2  # Converts -1, 1 to 0, 1
                loss = loss_function(y_pred, labels_adjusted)
                # Convert predictions back if necessary
                predicted_labels = (y_pred.squeeze() > 0.5) * 2 - 1  # Converts 0, 1 back to -1, 1
                # Binarize predictions based on a 0.5 threshold
                predicted_labels = (y_pred.squeeze() > 0.5).float()
                # Convert to -1 and 1
                predicted_labels = np.where(predicted_labels.cpu().numpy() == 0, -1, 1)
                # Ensure predicted_labels is treated as an array
                predicted_labels_array = np.atleast_1d(predicted_labels)
                predictions.extend(predicted_labels_array)
                val_loss += loss.item()
                actuals.extend(labels.tolist())
        val_loss /= len(val_loader)

        if saved_to_file == False:
            # Save validation results for analysis
            df_results = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
            csv_file = 'predictions_vs_actuals_movement_validation_test.csv'
            df_results.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
            saved_to_file == True

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_path_actual_price_movement)
            print(f'New best validation loss: {best_loss}, saving model.')
        else:
            patience_counter += 1
            print(f'No improvement in validation loss for {patience_counter} epochs.')

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    name_of_folder = f"{Pair}_{timeframe_str}_{start_date_test_training}_{end_date_test_training}"
    create_diagrams(training_data, name_of_folder)

is_evaluating_actual_price_movement = input("Do you want to evaluate the model for the actual price movement? (yes or no)").lower().strip() == 'yes'

if is_evaluating_actual_price_movement:
    sequence_length = 1
    start_date_evaluating = get_user_date_input("Please enter the start date for testing the trained model (YYYY-MM-DD): ")
    end_date_test_evaluating = get_user_date_input("Please enter the end date for testing the trained model (YYYY-MM-DD): ")

    strategy_data_out_of_sample = eur_usd_data[(eur_usd_data.index >= start_date_evaluating) & (eur_usd_data.index <= end_date_test_evaluating)]

    feature_columns = ['Previous Actual Price Movement'] 

    target_column = 'Actual Price Movement'

    test_loader = preprocess_data(strategy_data_out_of_sample, feature_columns, target_column, sequence_length)

    model_loaded = LSTM_Actual_Price_Movement_Prediction(input_size=len(feature_columns), hidden_layer_size=100, output_size=1)
    model_loaded.load_state_dict(torch.load(model_path_actual_price_movement))
    model_loaded.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for seq, labels in test_loader:
            y_test_pred = model_loaded(seq)
            # Binarize predictions based on a 0.5 threshold
            predicted_labels = (y_test_pred.squeeze() > 0.5).float()
            # Convert to -1 and 1
            predicted_labels = np.where(predicted_labels.cpu().numpy() == 0, -1, 1)
            # Ensure predicted_labels is treated as an array
            predicted_labels_array = np.atleast_1d(predicted_labels)
            predictions.extend(predicted_labels_array)
            # Ensure labels are also treated as an array before squeezing
            actuals.extend(np.atleast_1d(labels.cpu().numpy().squeeze()))

    # Create 'df_results' with the correctly aligned data
    df_results = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
    })

    # Convert 'Actual' column to integers
    df_results['Actual'] = df_results['Actual'].astype(int)

    df_results['Predicted'] = df_results['Predicted'].astype(int)

    df_results = df_results[(df_results['Actual'] != 0) & (df_results['Predicted'] != 0)]

    csv_file = 'predictions_vs_actuals_movement.csv'
    df_results.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))

    accuracy = accuracy_score(df_results['Actual'], df_results['Predicted'])
    precision = precision_score(df_results['Actual'], df_results['Predicted'], pos_label=1)
    recall = recall_score(df_results['Actual'], df_results['Predicted'], pos_label=1)
    f1 = f1_score(df_results['Actual'], df_results['Predicted'], pos_label=1)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    name_of_folder = f"{Pair}_{timeframe_str}_{start_date_evaluating}_{end_date_test_evaluating}"

    create_diagrams(strategy_data_out_of_sample, name_of_folder)

