import streamlit as st
import pickle


st.write("Hello, World!")

model_="rnn"
dataset_="2yr"
forecast_horizon=4

with open(f'pages/models/{model_}_{dataset_}_fh{forecast_horizon}.pickle', 'rb') as handle:
        params = pickle.load(handle)