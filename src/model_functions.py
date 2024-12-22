import sys
from pathlib import Path
import keras._tf_keras.keras.initializers as initializers
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, GRU, SimpleRNN, Dense, Input
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import L1
from typing import Optional

import time

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scikeras.wrappers import KerasRegressor
# from src.model_functions import build_model

import numpy as np

def build_model(
    model_,
    neurons,
    l1_reg,
    seed,
    n_steps,
    n_steps_ahead
):
    
    """
    Build a specified model (rnn, gru, or lstm) with the specified parameters.
    """
    model = Sequential()

    if model_.lower()=='rnn':
         model.add(
            SimpleRNN(
                units=neurons,
                activation='tanh',
                kernel_initializer=initializers.glorot_uniform(seed),
                bias_initializer=initializers.glorot_uniform(seed),
                recurrent_initializer=initializers.orthogonal(seed),
                kernel_regularizer=L1(l1_reg),
                input_shape=(n_steps, 1),
                unroll=False,
                stateful=False
                )
        )
    elif model_.lower()=='gru':
          model.add(
            GRU(
                units=neurons,
                activation='tanh',
                kernel_initializer=initializers.glorot_uniform(seed),
                bias_initializer=initializers.glorot_uniform(seed),
                recurrent_initializer=initializers.orthogonal(seed),
                kernel_regularizer=L1(l1_reg),
                input_shape=(n_steps, 1),
                unroll=False,
                stateful=False
                )
        )
    elif model_.lower()=='lstm':
           model.add(
            LSTM(
                units=neurons,
                activation='tanh',
                kernel_initializer=initializers.glorot_uniform(seed),
                bias_initializer=initializers.glorot_uniform(seed),
                recurrent_initializer=initializers.orthogonal(seed),
                kernel_regularizer=L1(l1_reg),
                input_shape=(n_steps, 1),
                unroll=False,
                stateful=False
                )
        )

    model.add(
        Dense(
            n_steps_ahead,
            kernel_initializer=initializers.glorot_uniform(seed),
            bias_initializer=initializers.glorot_uniform(seed),
            kernel_regularizer=L1(l1_reg)
        )
    )

    # compile model
    model.compile(
        loss='mean_absolute_error',
        optimizer='adam'
    )

    return model

def cross_val(params, batch_size, max_epochs, x_train_, y_train_, es, n_steps, n_steps_ahead):
    n_units = [5, 10, 20, 25, 30]
    l1_reg = [0.001, 0.01, 0.1]
    seed = 0  # You can adjust the seed value if necessary
    
    # A dictionary containing a list of values to be iterated through
    # for each parameter of the model included in the search
    param_grid = {'n_units': n_units, 'l1_reg': l1_reg}
    
    # A grid search is performed for each of the models
    for key in params.keys():
        print('Performing cross-validation. Model:', key)
        print(f'Training on a dataset of length {len(x_train_)}')
        
        # add start time
        start_time_cv = time.time()
        
        if key == 'rnn':
            model = KerasRegressor(
            model=build_model(model_='rnn', neurons=n_units, l1_reg=l1_reg, seed=seed, n_steps=n_steps, n_steps_ahead=n_steps_ahead),
            l1_reg=l1_reg,
            n_units=n_units,
            epochs=max_epochs, 
            batch_size=batch_size,
            verbose=2
        )
            
        elif key == 'gru':
            return None
        
        elif key == 'lstm':
            return None
        
        # Perform Grid Search
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid, 
            cv=TimeSeriesSplit(n_splits=4),
            verbose=2
        )
        
        grid_result = grid.fit(
            x_train_,
            y_train_,
            callbacks=[es]
        )
        
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
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

        params[key]['H'] = grid_result.best_params_['n_units']
        params[key]['l1_reg'] = grid_result.best_params_['l1_reg']
        params[key]['cv_time'] = end_time_cv - start_time_cv
