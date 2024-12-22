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

from keras._tf_keras.keras import initializers
from keras._tf_keras.keras import layers
from keras._tf_keras.keras import models
from keras._tf_keras.keras import regularizers
from keras._tf_keras.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor

from src.model_functions import build_model

import pickle
from src.ETL import Extract, Transform, Load

class Modeling:

    def __init__(self, batch_size: int, max_epochs: int, n_steps: int, forecast_horizon: int):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_steps = n_steps
        self.forecast_horizon = forecast_horizon

    def cross_validate(self, params: dict, param_grid: dict[str, list], es: EarlyStopping, dataset: str):

        load = Load(dataset_=dataset)
        train_test_val_dict = load.load_data()

        # Get the training data for cross-validation
        # The training data is already scaled and reformatted for input in nn models
        x_train_ = train_test_val_dict['train']['X_scaled']
        y_train_ = train_test_val_dict['train']['y_scaled']

        # Get the values from params
        n_units = param_grid.get("n_units", [])
        l1_reg = param_grid.get("l1_reg", [])

        # set seed as 0 for reproducibility of results
        seed = 0
        
        for key in params.keys():
            print('Performing cross-validation. Model:', key)

            # start recording the time
            start_time_cv = time.time()

            if key == 'rnn':
                model = KerasRegressor(
                            model = lambda n_units, l1_reg: build_model(
                                model_='rnn', neurons=n_units, 
                                l1_reg=l1_reg, seed=seed, 
                                n_steps=self.n_steps, n_steps_ahead=self.forecast_horizon
                                ),
                            l1_reg=l1_reg,
                            n_units=n_units,
                            epochs=self.max_epochs, 
                            batch_size=self.batch_size,
                            verbose=2
                        )

            elif key == 'gru':
                model = KerasRegressor(
                            model = lambda n_units, l1_reg: build_model(
                                model_='rnn', neurons=n_units, 
                                l1_reg=l1_reg, seed=seed, 
                                n_steps=self.n_steps, n_steps_ahead=self.forecast_horizon
                                ),
                            l1_reg=l1_reg,
                            n_units=n_units,
                            epochs=self.max_epochs, 
                            batch_size=self.batch_size,
                            verbose=2
                        )

            elif key == 'lstm':
                model = KerasRegressor(
                            model = lambda n_units, l1_reg: build_model(
                                model_='rnn', neurons=n_units, 
                                l1_reg=l1_reg, seed=seed, 
                                n_steps=self.n_steps, n_steps_ahead=self.forecast_horizon
                                ),
                            l1_reg=l1_reg,
                            n_units=n_units,
                            epochs=self.max_epochs, 
                            batch_size=self.batch_size,
                            verbose=2
                        )

            # perform grid search
            grid = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid, 
                        cv=TimeSeriesSplit(n_splits=4),
                        verbose=2
                        )
            grid_result = grid.fit(
                x_train_,y_train_, callbacks=[es]
                        )
            print(
                "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)
                )
            # Add end time
            end_time_cv = time.time()
        
            # Extract results
            means_ = grid_result.cv_results_['mean_test_score']
            stds_ = grid_result.cv_results_['std_test_score']
            params_ = grid_result.cv_results_['params']
            for mean, stdev, param_ in zip(means_, stds_, params_):
                print("%f (%f) with %r" % (mean, stdev, param_))
                
            # Save the results in the params dictionary
            params[key]['cv_results']['means_'] = means_
            params[key]['cv_results']['stds_'] = stds_
            params[key]['cv_results']['params_'] = params_

            # record the n_units and l1_reg for the best model and the time took for cross-validating
            params[key]['H'] = grid_result.best_params_['n_units']
            params[key]['l1_reg'] = grid_result.best_params_['l1_reg']
            params[key]['cv_time'] = end_time_cv - start_time_cv # records the 

            # save params
            with open(f'data/processed/{key}_{dataset}_fh{self.forecast_horizon}.pickle', 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, es: EarlyStopping, dataset_: str, model_: str):

        # read params:

        with open(f'data/processed/{model_}_{dataset_}_fh{self.forecast_horizon}.pickle', 'rb') as handle:
                params = pickle.load(handle)

        for key in params.keys():
            # set seed as 0 for reproducibility of results
            tf.random.set_seed(0)

            print('Training', key, 'model')


            # Load the staged train_val_test_dict 
            load = Load(dataset_=dataset_)
            train_val_test_dict = load.load_data()

            # Load the training, validation, and testing sets (already formatted and scaled)
            X_train_ = train_val_test_dict['train']['X_scaled']
            y_train_ = train_val_test_dict['train']['y_scaled']
            X_val_ = train_val_test_dict['val']['X_scaled']
            y_val_ = train_val_test_dict['val']['y_scaled']
            X_test_ = train_val_test_dict['test']['X_scaled']
            y_test_ = train_val_test_dict['test']['y_scaled']

            # scalers (instances of StandardScaler())
            scaler_train = train_val_test_dict['train']['scaler']
            scaler_val = train_val_test_dict['val']['scaler']
            scaler_test = train_val_test_dict['test']['scaler']
            # training
            start_train = time.time()

            model = build_model(model_=key, neurons = params[key]['H'], l1_reg=params[key]['l1_reg'], seed=0, n_steps = self.n_steps, n_steps_ahead=self.forecast_horizon)
            print(f"Training {key} model with: {params[key]['H']} neurons and {params[key]['l1_reg']} l1 reg.")

            # hardcoded epochs=750 (more than the 250 max epochs) to allow the best model to learn more about the data
            model.fit(X_train_, y_train_, epochs=750, validation_data=(X_val_, y_val_),
                    batch_size=self.batch_size, callbacks=[es], shuffle=False)
            
            end_train = time.time()

            params[key]['model'] = model
            params[key]['train_time'] = end_train - start_train


        for key in params.keys():
            model = params[key]['model']

            params[key]['pred_train'] = model.predict(X_train_, verbose=1)
            params[key]['MSE_train'] = mean_squared_error(y_train_, params[key]['pred_train'])

            params[key]['pred_val'] = model.predict(X_val_, verbose=1)
            params[key]['MSE_val'] = mean_squared_error(y_val_, params[key]['pred_val'])
            
            params[key]['pred_test'] = model.predict(X_test_, verbose=1) 
            params[key]['MSE_test'] = mean_squared_error(y_test_, params[key]['pred_test'])

            params[key]['pred_train_scaled'] = scaler_train.inverse_transform(params[key]['pred_train'])
            params[key]['y_train_scaled'] = scaler_train.inverse_transform(y_train_)

            params[key]['pred_val_scaled'] = scaler_val.inverse_transform(params[key]['pred_val'])
            params[key]['y_val_scaled'] = scaler_val.inverse_transform(y_val_)

            params[key]['pred_test_scaled'] = scaler_test.inverse_transform(params[key]['pred_test'])
            params[key]['y_test_scaled'] = scaler_test.inverse_transform(y_test_)


            # Record the MSE, MAE, and R2 metrics in the pickle file
            params[key]['MSE_train_scaled'] = mean_squared_error(params[key]['pred_train_scaled'], params[key]['y_train_scaled'])
            params[key]['MSE_val_scaled'] = mean_squared_error(params[key]['pred_val_scaled'], params[key]['y_val_scaled'])
            params[key]['MSE_test_scaled'] = mean_squared_error(params[key]['pred_test_scaled'], params[key]['y_test_scaled'])

            params[key]['MAE_train_scaled'] = mean_absolute_error(params[key]['pred_train_scaled'], params[key]['y_train_scaled'])
            params[key]['MAE_val_scaled'] = mean_absolute_error(params[key]['pred_val_scaled'],params[key]['y_val_scaled'])
            params[key]['MAE_test_scaled'] = mean_absolute_error(params[key]['pred_test_scaled'],params[key]['y_test_scaled'])

            params[key]['R2_train_scaled'] = r2_score(params[key]['pred_train_scaled'], params[key]['y_train_scaled'])
            params[key]['R2_val_scaled'] = r2_score(params[key]['pred_val_scaled'], params[key]['y_val_scaled'])
            params[key]['R2_test_scaled'] = r2_score(params[key]['pred_test_scaled'], params[key]['y_test_scaled'])
            

            # Record the mean and stadard dev for StandardScaler(), to be used as scaler for user-input values
            for split in ['train', 'val', 'test']:
                for stat in ['mean', 'std']:
                    params[key][f'scaler_{split}_{stat}'] = train_val_test_dict[split][f'scaler_{stat}']
            
            with open(f'models/{dataset_}_models_{key}.pkl', 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'dags/streamlit/pages/models/{dataset_}_models_{key}.pkl', 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)