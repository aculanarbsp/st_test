import numpy as np
import pandas as pd
import statsmodels.api as sm # library for statistical computations

from src.functions import split_data, reformat_to_arrays, scale_the_data

import pickle

from datetime import datetime

class Extract:
    # Get the data from the source

    def __init__(self, dataset_:str):
        # Note: dataset_ refers to the TENOR of the bond (i.e. 2-year or 10-year)
        self.dataset_ = dataset_

    def get_data(self):
        # Validate the tenor input to ensure it's a valid format
        if not isinstance(self.dataset_, str):
            raise ValueError("Tenor must be a string.")
        if self.dataset_.upper() not in ['2YR', '10YR']:  # you can adjust this based on your valid options
            raise ValueError("Invalid tenor. Only '2YR' or '10YR' are allowed for now.")

        df = pd.read_csv(
            f'./data/raw/bond_yields - USGG{self.dataset_.upper()}.csv',
            index_col=0 # will read the first column as the index
            )

        df.index = pd.to_datetime(
            df.index
            # infer_datetime_format=True
            )
    
        df = df.loc['2005-01-01':'2024-08-31'] # get only from year 2005

        return df

class Transform:

    def __init__(self, dataset_:str):
        # Note: dataset_ refers to the TENOR of the bond (i.e. 2-year or 10-year)
        self.dataset_ = dataset_

    # returns three dataframes: the train set, validation set, and the test set
    def train_val_test_split(self, train_split: float, val_split: float):

        extract = Extract(self.dataset_)
        df = extract.get_data()

        train, val, test = split_data(df, train_pct=train_split, val_pct=val_split)
        return train, val, test
    
    # note: remember to get the lag of the training data only
    def get_lags(self, significance: float):
        # significance can be between 95% 96% 99%

        extract = Extract(self.dataset_)
        df = extract.get_data()

        # Get stationarity:

        adf, p, usedlag, nobs, cvs, aic = sm.tsa.stattools.adfuller(df)

        # from the usedlag, get pacf
        pacf = sm.tsa.stattools.pacf(df['yield'], nlags=usedlag)

        T = len(df['yield'])
        tau_h = 2.58 # 99% significance
        sig_test = lambda tau_h: np.abs(tau_h) > 2.58/np.sqrt(T)


        # We set a minimum of 7 for the lags
        # for i in range(len(pacf)):
        #     if (sig_test(pacf[i]) == False) & ((i-1) >= 7):
        #         n_steps = i - 1
        #         print('n_steps set to', n_steps)
        #         break
        
        # hardcoded for now:
        n_steps = 15

        return n_steps
    
    # Uses n_steps (from get_lags) and the desired forecasting horizon to split and reformat
    # the data for model selection

    def split_and_reformat(self, train_split:int, val_split:int, forecast_horizon: int):
        
        extract = Extract(self.dataset_)
        df = extract.get_data()

        train, val, test = self.train_val_test_split(train_split=train_split, val_split=val_split)

        # determine the lag (n_steps)

        n_steps = self.get_lags(2.58)

        train_scaled, train_mean, train_std, scaler_for_training_fitted_ = scale_the_data(train)
        val_scaled, val_mean, val_std, scaler_for_val_fitted_= scale_the_data(val)
        test_scaled, test_mean, test_std, scaler_for_test_fitted_ = scale_the_data(test)

        X_train_scaled_, y_train_scaled_ = reformat_to_arrays(train_scaled, n_steps=n_steps, n_steps_ahead=forecast_horizon)
        X_val_scaled_, y_val_scaled_ = reformat_to_arrays(val_scaled, n_steps=n_steps, n_steps_ahead=forecast_horizon)
        X_test_scaled_, y_test_scaled_ = reformat_to_arrays(test_scaled, n_steps=n_steps, n_steps_ahead=forecast_horizon)
        # print(f"The length of X_train is {len(X_train_scaled_)} and the length of y_train is {len(y_train_scaled_)}")

        train_val_test_dict = {
            'train': 
                {'dataframe': train, 'scaler': scaler_for_training_fitted_, 
                'scaler_mean': train_mean, 'scaler_std': train_std, 
                'X_scaled': X_train_scaled_, 'y_scaled': y_train_scaled_},
            'val': 
                {'dataframe': val, 'scaler': scaler_for_val_fitted_, 
                'scaler_mean': val_mean, 'scaler_std': val_std,
                'X_scaled': X_val_scaled_, 'y_scaled': y_val_scaled_},
            'test': 
                {'dataframe': test, 'scaler': scaler_for_test_fitted_, 
                'scaler_mean': test_mean, 'scaler_std': test_std, 
                'X_scaled': X_test_scaled_, 'y_scaled': y_test_scaled_}
            }
        
        # now = datetime.now()
        # dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")

        with open(f'data/staging/train_val_test_{self.dataset_}.pickle', 'wb') as handle:
            pickle.dump(train_val_test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Loads the data from the staging folder. The staging folder contains data for the consumption of model selection and training phases.
class Load:

    def __init__(self, dataset_:str):
        # Note: dataset_ refers to the TENOR of the bond (i.e. 2-year or 10-year)
        self.dataset_ = dataset_

    def load_data(self):

        # reads the staged data (in the form of pickle file)
        with open(f"data/staging/train_val_test_{self.dataset_}.pickle", "rb") as file:
            train_val_test_dict = pickle.load(file)

        return train_val_test_dict
