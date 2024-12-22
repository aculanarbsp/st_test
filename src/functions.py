import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm # library for statistical computations
import tensorflow as tf
from datetime import timedelta

import time

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from keras import initializers
from keras import layers # Dense, LSTM, GRU, SimpleRNN
from keras import models # Sequential, load_model
from keras import regularizers # l1, l2
from keras import callbacks # EarlyStopping
from scikeras.wrappers import KerasRegressor
# from kerasreg_custom import CustomKerasRegressor

import pickle
import gc
from functools import partial

unroll_setting = False

# convert history into inputs and outputs
def reformat_to_arrays(data, n_steps, n_steps_ahead):
    X, y = list(), list()
    in_start = 0
    
    for _ in range(len(data)):
        in_end = in_start + n_steps
        out_end = in_end + n_steps_ahead
        
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
        
    return np.array(X), np.array(y)


# This function splits the data into training, validation, and test sets
# df = dataframe to split
# train_pct = The percentage of the data we want to be treated as training set. (e.g. if we want 60%, then training_size = 0.6)
# val_pct = similar to training_size, but for validation set
def split_data(df, train_pct, val_pct):

    # Check if the sum of train_pct and val_pct exceeds 1 
    if train_pct + val_pct > 1:
        raise ValueError(f"The sum of train_pct and val_pct should not exceed 1.")

     
    # Define the split ratios
    train_size = int(len(df) * train_pct) 
    val_size = int(len(df) * val_pct)

    # We get the train size by subtracting the train_size and val_size
    test_size = len(df) - train_size - val_size 

    # Split the data
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    # returns numpy arrays of train, val, and test
    return train_data, val_data, test_data
	
# Scales the given data using StandardScaler()
# Also saves the mean and standard deviation computed from fitting the StandardScaler() with the given data.
# We save the mean and standard deviation because the input for models are scaled data, therefore we will 
# scale the input data before inputting in the models. Outputs are also scaled and we inverse_scale them to
# get the proper value.
def scale_the_data(data):
    scaler = StandardScaler()

    scaler_fit = scaler.fit(data)
    mean = scaler_fit.mean_
    std = scaler_fit.scale_

    transformed_data = scaler_fit.transform(data)

    return transformed_data, mean, std, scaler_fit


def plotPACF(df: pd.DataFrame, filename: str):

    use_features = "yield"

    adf, p, usedlag, nobs, cvs, aic = sm.tsa.stattools.adfuller(df[use_features])
    adf_results_string = 'ADF: {}\np-value: {},\nN: {}, \ncritical values: {}'
    # print(adf_results_string.format(adf, p, nobs, cvs))

    pacf = sm.tsa.stattools.pacf(df[use_features], nlags=usedlag)

    T = len(df[use_features])

    z_score = 2.58 # 99% confidence interval

    plt.plot(pacf, label='pacf')
    plt.plot([z_score/np.sqrt(T)]*30, label='99% confidence interval (upper)')
    plt.plot([-z_score/np.sqrt(T)]*30, label='99% confidence interval (lower)')
    plt.xlabel('number of lags')
    plt.legend()
    plt.show()
    plt.savefig(f"figures/{filename}.png", dpi=1000)


def plotAveragedPrediction(predictions, ground_truth):
    
    # Example: Overlapping predictions (4-day forecasts) and ground truth (as continuous values)
    all_predictions = predictions
    ground_truth_4_arrays = ground_truth

    # Convert ground truth into a single continuous array
    ground_truth_continuous = []
    for i in range(len(ground_truth_4_arrays) - 1):
        ground_truth_continuous.append(ground_truth_4_arrays[i, 0])
        ground_truth_continuous.extend(ground_truth_4_arrays[-1])
    
    ground_truth_continuous = np.array(ground_truth_continuous)

    # Time axis for ground truth
    time_axis_truth = np.arange(len(ground_truth_continuous))

    # Initialize arrays for averaged predictions
    time_series_length = len(all_predictions) + 3  # Include overlapping days
    predicted_full = np.zeros(time_series_length)
    overlap_count = np.zeros(time_series_length)  # To average overlapping predictions

    # Fill the arrays for predictions
    for i, forecast in enumerate(all_predictions):
        predicted_full[i:i+4] += forecast
        overlap_count[i:i+4] += 1

    # Average overlapping predictions
    predicted_full /= overlap_count

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot ground truth as a single continuous line
    plt.plot(time_axis_truth, ground_truth_continuous, label="Ground Truth", linestyle="-", marker="o", color="black")

    # Plot the averaged predictions
    time_axis_predictions = np.arange(time_series_length)
    plt.plot(time_axis_predictions, predicted_full, label="Averaged Predictions", linestyle="--", color="blue")

    # Finalize the plot
    plt.title("Averaged Predictions vs Ground Truth")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

def predictModel():
    return None

# reads the pickle file for the given model and dataset
# dataset: 2YR or 10YR
# model: RNN, GRU, LSTM
def load_params(dataset_, model_):

    with open(f"app/pages/models/{dataset_.upper()}_models_{model_.lower()}.pkl", "rb") as file:
        params_model = pickle.load(file)

    return params_model