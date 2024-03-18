import pandas as pd  # Import pandas for data manipulation and analysis
import datetime  # Import datetime for working with dates and times
import MetaTrader5 as mt5  # Import the MetaTrader5 module to interact with the MetaTrader 5 terminal
from datetime import datetime, timedelta  # Import specific classes from datetime for convenience
import os  # Import os module for interacting with the operating system
import pytz  # Import pytz for timezone calculations
import pandas_ta as ta  # Import pandas_ta for technical analysis indicators on pandas DataFrames

import torch  # Import torch for deep learning tasks
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting graphs
import matplotlib.pyplot as plt  # This is a duplicate import and can be removed
torch.distributions.Categorical  # Access the Categorical distribution class from PyTorch's distributions module

from sklearn.metrics import accuracy_score  # Import accuracy_score for model evaluation
from sklearn.metrics import precision_score  # Import precision_score for model evaluation
from sklearn.metrics import recall_score  # Import recall_score for model evaluation
from sklearn.metrics import f1_score  # Import f1_score for model evaluation

import torch  # This is a duplicate import and can be removed
import torch.nn as nn  # Import torch.nn for building neural network layers
from torch.utils.data import DataLoader, TensorDataset  # Import DataLoader and TensorDataset for handling datasets in PyTorch
import numpy as np  # Import numpy for numerical operations
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for feature scaling

# Your MetaTrader 5 login details
account_number = 1058146570  # Replace with your account number
password = 'ieasDILwkA8*'  # Replace with your password
server_name ='FTMO-Demo'  # Replace with your server name

def mt5_login(account_number, password, server_name):
    # Initialize MT5 connection; if unsuccessful, print error and exit
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    # Log into the specified account using the provided credentials; if unsuccessful, print error, shutdown MT5 and exit
    authorized = mt5.login(account_number, password=password, server=server_name)
    if not authorized:
        print("login failed, error code =", mt5.last_error())
        mt5.shutdown()
        quit()

def fetch_and_prepare_fx_data_mt5(symbol, timeframe_str, start_date, end_date):
    # Log in to MT5 account before fetching data
    mt5_login(account_number, password, server_name)

    # Set the time zone for data retrieval
    timezone = pytz.timezone("Europe/Berlin")

    # Convert start and end dates to datetime objects, setting timezone
    start_date = start_date.replace(tzinfo=timezone)
    end_date = end_date.replace(hour=23, minute=59, second=59, tzinfo=timezone)

    # Map string timeframe to MT5 timeframe constants
    timeframes = {
        '1H': mt5.TIMEFRAME_H1,
        'DAILY': mt5.TIMEFRAME_D1,
        # Add more mappings as needed
    }

    # Retrieve the correct timeframe constant using the provided string key
    timeframe = timeframes.get(timeframe_str)
    if timeframe is None:
        print(f"Invalid timeframe: {timeframe_str}")
        mt5.shutdown()
        return None

    # Fetch the rate data for the specified symbol, timeframe, and date range
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    # If no rates were retrieved, print error, shutdown MT5 and return None
    if rates is None:
        print("No rates retrieved, error code =", mt5.last_error())
        mt5.shutdown()
        return None
    
    # Convert the retrieved rates into a pandas DataFrame
    rates_frame = pd.DataFrame(rates)
    # Convert the 'time' column to datetime format
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

    # Set the 'time' column as the index of the DataFrame
    rates_frame.set_index('time', inplace=True)
    # Ensure the index is in the proper datetime format
    rates_frame.index = pd.to_datetime(rates_frame.index, format="%Y-%m-%d %H:%M:%S")
    
    # Shutdown MT5 connection after data retrieval
    mt5.shutdown()
    
    # Return the prepared DataFrame
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

# Fetch Historical data from 1971 to present day
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

    # Add column for Previous Close Price which is just the close price column shifted up by 1
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
    # Define the directory path for saving graphs, incorporating the specified folder name
    graphs_dir = f'Graphs, {name_of_folder}'
    # Create the directory if it doesn't exist, without raising an error if it already exists
    os.makedirs(graphs_dir, exist_ok=True)

    plt.ioff()  # Turn off interactive mode to prevent figures from displaying

    # Create a new figure with specified size
    plt.figure(figsize=(14, 7))
    # Plot closing prices
    plt.plot(data.index, data['close'], label='Close Price')
    # Plot 50-day Simple Moving Average (SMA)
    plt.plot(data.index, data['SMA_50'], label='50-Day SMA', alpha=0.75)
    # Plot 200-day SMA
    plt.plot(data.index, data['SMA_200'], label='200-Day SMA', alpha=0.75)
    # Plot 50-day Exponential Moving Average (EMA) in cyan
    plt.plot(data.index, data['EMA_50'], label='50-Day EMA', color='cyan', alpha=0.75)
    # Plot 200-day EMA in magenta
    plt.plot(data.index, data['EMA_200'], label='200-Day EMA', color='magenta', alpha=0.75)
    
    # Plot 20-day SMA for Bollinger Bands in orange
    plt.plot(data.index, data['SMA_20'], label='20-Day SMA', color='orange', alpha=0.75)
    # Plot the upper Bollinger Band in grey, dashed line
    plt.plot(data.index, data['Upper Band'], label='Upper Bollinger Band', linestyle='--', color='grey')
    # Plot the lower Bollinger Band in grey, dashed line
    plt.plot(data.index, data['Lower Band'], label='Lower Bollinger Band', linestyle='--', color='grey')
    # Fill between the upper and lower Bollinger Bands with grey color at 10% opacity
    plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.1)

    plt.title('EUR/USD with SMAs, EMAs, and Bollinger Bands')
    # Save the figure to the specified path in the graphs directory
    plt.savefig(f'{graphs_dir}/price_smas_emas_bollinger_bands.png')
    plt.legend()
    plt.close()  # Close the figure to free memory

    # Create a new figure for RSI plot with specified size
    plt.figure(figsize=(14, 7))
    # Plot RSI values in red
    plt.plot(data.index, data[f'RSI_9'], label='RSI', color='red', alpha=0.75)
    # Plot overbought threshold line at RSI 70
    plt.axhline(70, color='green', linestyle='dotted', label='Overbought (70)')
    # Plot oversold threshold line at RSI 30
    plt.axhline(30, color='blue', linestyle='dotted', label='Oversold (30)')

    plt.title('RSI with Overbought and Oversold Levels')
    plt.legend()
    # Save the RSI plot to the graphs directory
    plt.savefig(f'{graphs_dir}/rsi.png')
    plt.close()  # Close the figure

    # Create a new figure for displaying the table of the last 20 rows
    plt.figure(figsize=(14, 7))
    # Prepare data for the table by selecting the last 20 rows and resetting the index
    table_data = data.tail(20).reset_index()
    # Select specific columns for the table
    table_data = table_data[['time', 'close', 'Previous Close', 'Actual Price Movement', 'Previous Actual Price Movement', 'EMA_200',]]

    ax_table = plt.subplot(111, frame_on=False)  # Create a subplot without a visible frame
    ax_table.xaxis.set_visible(False)  # Hide the x-axis
    ax_table.yaxis.set_visible(False)  # Hide the y-axis
    # Create the table in the center of the figure
    table = plt.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Set font size for the table
    table.scale(1.2, 1.2)  # Scale the table size
    plt.title(f'Last 20 Rows of {Pair} Data')  # Title of the figure might need correction as 'Pair' is undefined in this snippet
    # Save the table figure to the graphs directory
    plt.savefig(f'{graphs_dir}/last_20_rows_table.png')
    plt.close()  # Close the figure to free memory

    # Create a new figure for the MACD and Signal Line plot
    plt.figure(figsize=(14, 7))
    # Plot MACD values in blue
    plt.plot(data.index, data['MACD'], label='MACD', color='blue')
    # Plot Signal Line values in red
    plt.plot(data.index, data['Signal_Line'], label='Signal Line', color='red')
    plt.title('MACD and Signal Line')
    plt.legend()
    # Save the MACD and Signal Line plot to the graphs directory
    plt.savefig(f'{graphs_dir}/macd_signal_line.png')
    plt.close()  # Close the figure to free memory

def create_inout_sequences(input_data, target_data, tw):
    inout_seq = []  # Initialize an empty list for the input-output sequences
    L = len(input_data)  # Determine the length of the input data
    for i in range(L-tw):  # Loop over the input data with a window of size 'tw'
        # Create a tensor from a slice of input data for the current sequence
        train_seq = torch.tensor(input_data[i:i+tw], dtype=torch.float32)
        # Create a tensor for the label (target value) corresponding to the end of the current sequence
        train_label = torch.tensor(target_data[i+tw], dtype=torch.float32)
        # Append the sequence and its corresponding label as a tuple to the inout_seq list
        inout_seq.append((train_seq, train_label.unsqueeze(-1)))  # Ensure the label is correctly shaped
    return inout_seq  # Return the list of tuples

def preprocess_data(data, feature_columns, target_column, sequence_length):
    # Initialize a MinMaxScaler to scale the feature columns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Scale the feature columns and assign the result
    data_scaled_features = scaler.fit_transform(data[feature_columns])
    
    # Initialize another MinMaxScaler for the target column
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    # Scale the target column, ensure it's a flat array for compatibility
    data_scaled_target = target_scaler.fit_transform(data[[target_column]].values).flatten()
    
    # Create input-output sequences using the scaled features and target
    sequences = create_inout_sequences(data_scaled_features, data_scaled_target, sequence_length)

    # Convert each sequence in sequences to a tensor, and collect them
    sequence_tensors = [s[0].clone().detach() for s in sequences]
    # Similarly, convert each target in sequences to a tensor
    target_tensors = [s[1].clone().detach() for s in sequences]

    # Create a TensorDataset from the sequence and target tensors
    dataset = TensorDataset(torch.stack(sequence_tensors), torch.stack(target_tensors))
    
    # Return a DataLoader for the dataset, with specified batch size and shuffling
    return DataLoader(dataset, batch_size=64, shuffle=True)
   
class LSTM_Actual_Price_Movement_Prediction(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()  # Initialize the superclass
        self.hidden_layer_size = hidden_layer_size  # Set the hidden layer size attribute
        # Initialize an LSTM layer with specified input size, hidden layer size, and batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        # Initialize a linear layer to map from hidden layer size to output size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Initialize a Sigmoid activation function

    def forward(self, input_seq):
        # Initialize the initial hidden state and cell state for the LSTM
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size, dtype=torch.float32)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size, dtype=torch.float32)
        # Pass the input sequence through the LSTM layer
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        # Pass the output of the LSTM's last time step through the linear layer and then the sigmoid activation
        predictions = self.sigmoid(self.linear(lstm_out[:, -1, :]))  # Apply sigmoid
        return predictions  # Return the final predictions

# Boolean to confirm if we have saved a file or not    
saved_to_file = False

# Apply technical indicators to the data using the 'calculate_indicators' function
eur_usd_data = calculate_indicators(eur_usd_data)

# Define the path for saving the trained model, incorporating the currency pair and timeframe
model_path_actual_price_movement = f"lstm_model_{Pair}_{timeframe_str}_actual_price_movement.pth"

# Ask the user if they want to train the model, converting the response to lowercase and stripping whitespace, then checking if it's 'yes'
is_training_actual_price_movement = input("Do you want to train the model for the actual price movement? (yes or no): ").lower().strip() == 'yes'

# If the user wants to train the model, proceed with the following steps
if is_training_actual_price_movement:
    # Get user input for the start date of training data
    start_date_test_training = get_user_date_input("Please enter the start date for training data (YYYY-MM-DD): ")
    # Get user input for the end date of training data
    end_date_test_training = get_user_date_input("Please enter the end date for training data (YYYY-MM-DD): ")

    # Filter the EUR/USD data for the specified training period using start and end dates
    strategy_data_in_sample = eur_usd_data[(eur_usd_data.index >= start_date_test_training) & (eur_usd_data.index <= end_date_test_training)]

    # Define the feature columns to be used in the model
    feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'EMA_9', 'EMA_21', 'SMA_20', 'Upper Band', 'Lower Band', 'Lower Band Scalping', 'Middle Band Scalping', 'Upper Band Scalping', 'MACD', 'Signal_Line', 'RSI_9', 'RSI_5 Scalping', 'Stoch_%K', 'Stoch_%D', 'Previous Close','Previous Actual Price Movement']  # Adjust as needed
    target_column = 'Actual Price Movement'  # Define the target column for prediction

    # Set the sequence length, indicating how many days of data will be used to predict the next day's movement
    sequence_length = 1  # The number of past days to consider for predicting the next day's close price

    # Calculate the index at which to split the data into training and validation sets, using 80% of data for training
    split_idx = int(len(strategy_data_in_sample) * 0.8)

    # Split the data into training and validation sets using the calculated index
    training_data = strategy_data_in_sample.iloc[:split_idx]
    validation_data = strategy_data_in_sample.iloc[split_idx:]

    # Preprocess the training and validation data, creating DataLoader objects for each
    train_loader = preprocess_data(training_data, feature_columns, target_column, sequence_length)
    val_loader = preprocess_data(validation_data, feature_columns, target_column, sequence_length)
    
    # Initialize the LSTM model with specified input size, hidden layer size, and output size
    model = LSTM_Actual_Price_Movement_Prediction(input_size=len(feature_columns), hidden_layer_size=100, output_size=1)
    # Initialize the loss function as Binary Cross Entropy Loss
    loss_function = nn.BCELoss()
    # Initialize the optimizer as Adam, with learning rate 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set up early stopping parameters: patience (number of epochs to wait for improvement) and initial best loss
    patience = 50
    patience_counter = 0
    best_loss = float('inf')

    # Ask the user for the desired number of epochs for training
    epochs = int(input("Please input the number of epochs you would like to run: "))

    # Loop through each epoch for training and validation
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        # Initialize variable to accumulate training loss
        train_loss = 0
        # Loop through each batch in the training DataLoader
        for seq, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            # Get predictions from the model for the current batch
            y_pred = model(seq)
            # Remove any unnecessary dimensions from the predictions
            y_pred = torch.squeeze(y_pred)
            # Ensure labels are correctly shaped
            labels = torch.squeeze(labels)
            # Adjust labels to match the expected format for BCELoss
            labels_adjusted = (labels + 1) / 2  # Converts -1, 1 to 0, 1
            # Calculate the loss between predictions and adjusted labels
            loss = loss_function(y_pred, labels_adjusted)
            # Perform backpropagation
            loss.backward()
            # Update model weights
            optimizer.step()
            # Accumulate the loss for this batch
            train_loss += loss.item()
        # Calculate average training loss for the epoch
        train_loss /= len(train_loader)
        
        # Validation phase begins
        val_loss = 0
        # Set the model to evaluation mode
        model.eval()
        # Initialize lists to store actual and predicted labels
        actuals, predictions = [], []
        # Temporarily disable gradient computation
        with torch.no_grad():
            # Loop through each batch in the validation DataLoader
            for seq, labels in val_loader:
                # Get predictions from the model for the current batch
                y_pred = model(seq)
                # Remove any unnecessary dimensions from the predictions
                y_pred = torch.squeeze(y_pred)
                # Ensure labels are correctly shaped
                labels = torch.squeeze(labels)
                # Adjust labels to match the expected format for BCELoss
                labels_adjusted = (labels + 1) / 2  # Converts -1, 1 to 0, 1
                # Calculate the loss between predictions and adjusted labels
                loss = loss_function(y_pred, labels_adjusted)
                # Optionally convert predictions back to original format
                predicted_labels = (y_pred.squeeze() > 0.5) * 2 - 1  # Converts 0, 1 back to -1, 1
                # Add predicted labels to the predictions list
                predictions.extend(predicted_labels)
                # Accumulate the loss for this batch
                val_loss += loss.item()
                # Add actual labels to the actuals list
                actuals.extend(labels.tolist())
        # Calculate average validation loss for the epoch
        val_loss /= len(val_loader)

        # Check if this is the first time validation results are being saved
        if saved_to_file == False:
            # Ensure that all values in 'predictions' are plain numbers, not tensor objects
            predictions = [p.item() if hasattr(p, 'item') else p for p in predictions]
            # Now, create the DataFrame
            df_results = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
            # Continue with CSV saving as before
            csv_file = 'predictions_vs_actuals_movement_validation_test.csv'
            df_results.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
            # Ensure the saved_to_file flag is correctly updated
            saved_to_file = True

        # Print training and validation loss for the current epoch
        #print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
        
        # Check for improvement in validation loss
        if val_loss < best_loss:
            # Update best loss to current validation loss
            best_loss = val_loss
            # Reset patience counter
            patience_counter = 0
            # Save the current model as the best model
            torch.save(model.state_dict(), model_path_actual_price_movement)
            # Print message indicating new best validation loss
            #print(f'New best validation loss: {best_loss}, saving model.')
        else:
            # Increment patience counter if no improvement
            patience_counter += 1
            # Print message indicating no improvement
            #print(f'No improvement in validation loss for {patience_counter} epochs.')

        # Check if early stopping criteria are met
        if patience_counter >= patience:
            # Print message indicating early stopping
            #print("Early stopping triggered.")
            # Break out of the loop to stop training
            break

    # Define the name of the folder where diagrams will be saved, incorporating relevant details
    name_of_folder = f"{Pair}_{timeframe_str}_{start_date_test_training}_{end_date_test_training}"
    # Call the function to create diagrams for the training data
    create_diagrams(training_data, name_of_folder)

# Ask user if they want to evaluate the model for actual price movement, and store the boolean result
is_evaluating_actual_price_movement = input("Do you want to evaluate the model for the actual price movement? (yes or no): ").lower().strip() == 'yes'

# Proceed with evaluation if the user responded with 'yes'
if is_evaluating_actual_price_movement:
    # Set the sequence length for the LSTM input
    sequence_length = 1
    # Get user input for the start date of the evaluation period
    start_date_evaluating = get_user_date_input("Please enter the start date for testing the trained model (YYYY-MM-DD): ")
    # Get user input for the end date of the evaluation period
    end_date_test_evaluating = get_user_date_input("Please enter the end date for testing the trained model (YYYY-MM-DD): ")

    # Filter the EUR/USD data to the specified evaluation period
    strategy_data_out_of_sample = eur_usd_data[(eur_usd_data.index >= start_date_evaluating) & (eur_usd_data.index <= end_date_test_evaluating)]

    # Define the features to be used by the model
    feature_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'EMA_9', 'EMA_21', 'SMA_20', 'Upper Band', 'Lower Band', 'Lower Band Scalping', 'Middle Band Scalping', 'Upper Band Scalping', 'MACD', 'Signal_Line', 'RSI_9', 'RSI_5 Scalping', 'Stoch_%K', 'Stoch_%D', 'Previous Close','Previous Actual Price Movement']

    # Specify the target column for prediction
    target_column = 'Actual Price Movement'

    # Preprocess the data and create a DataLoader for testing
    test_loader = preprocess_data(strategy_data_out_of_sample, feature_columns, target_column, sequence_length)

    # Initialize the model with the correct input size, hidden layer size, and output size
    model_loaded = LSTM_Actual_Price_Movement_Prediction(input_size=len(feature_columns), hidden_layer_size=100, output_size=1)
    # Load the previously trained model weights
    model_loaded.load_state_dict(torch.load(model_path_actual_price_movement))
    # Set the model to evaluation mode
    model_loaded.eval()

    # Initialize lists to store predictions and actual labels
    predictions = []
    actuals = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate through batches in the DataLoader
        for seq, labels in test_loader:
            # Get predictions from the model
            y_test_pred = model_loaded(seq)
            # Binarize predictions based on a 0.5 threshold
            predicted_labels = (y_test_pred.squeeze() > 0.5).float()
            # Convert predictions to -1 and 1 for comparison
            predicted_labels = np.where(predicted_labels.cpu().numpy() == 0, -1, 1)
            # Ensure predictions are treated as an array
            predicted_labels_array = np.atleast_1d(predicted_labels)
            # Extend the predictions list with the current batch's predictions
            predictions.extend(predicted_labels_array)
            # Extend the actuals list with the current batch's labels, ensuring they're treated as an array
            actuals.extend(np.atleast_1d(labels.cpu().numpy().squeeze()))

    # Create a DataFrame with the actual and predicted labels
    df_results = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
    })

    # Convert both 'Actual' and 'Predicted' columns to integers for scoring
    df_results['Actual'] = df_results['Actual'].astype(int)
    df_results['Predicted'] = df_results['Predicted'].astype(int)

    # Filter out rows where both 'Actual' and 'Predicted' are not 0 to avoid skewing metrics
    df_results = df_results[(df_results['Actual'] != 0) & (df_results['Predicted'] != 0)]

    # Define the CSV file path for saving results
    csv_file = 'predictions_vs_actuals_movement.csv'
    # Save the DataFrame to CSV, appending if the file exists and adding header if it doesn't
    df_results.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))

    # Calculate evaluation metrics for the predictions
    accuracy = accuracy_score(df_results['Actual'], df_results['Predicted'])
    precision = precision_score(df_results['Actual'], df_results['Predicted'], pos_label=1)
    recall = recall_score(df_results['Actual'], df_results['Predicted'], pos_label=1)
    f1 = f1_score(df_results['Actual'], df_results['Predicted'], pos_label=1)

    # Convert metrics to percentages for easier understanding
    accuracy_percent = accuracy * 100
    precision_percent = precision * 100
    recall_percent = recall * 100
    f1_percent = f1 * 100

    # Print the calculated metrics with brief explanations
    print(f"Accuracy: {accuracy_percent:.2f}% - This metric shows the overall correctness of the model across all classes. A higher accuracy means the model made more correct predictions out of all predictions made.")

    print(f"Precision: {precision_percent:.2f}% - Precision reflects the model's ability to correctly predict positive observations out of all observations it predicted as positive. It is crucial in situations where the cost of a false positive is high.")

    print(f"Recall: {recall_percent:.2f}% - Recall (or Sensitivity) measures the model's ability to detect positive observations from the actual positives available. It's important when the concern is to minimize false negatives.")

    print(f"F1 Score: {f1_percent:.2f}% - The F1 Score is the harmonic mean of precision and recall. It provides a balance between them, useful when you want to take both false positives and false negatives into account.")

    # Define the folder name for saving any diagrams, incorporating relevant details
    name_of_folder = f"{Pair}_{timeframe_str}_{start_date_evaluating}_{end_date_test_evaluating}"

    # Generate and save diagrams for the out-of-sample data
    create_diagrams(strategy_data_out_of_sample, name_of_folder)


