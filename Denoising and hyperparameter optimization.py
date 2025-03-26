
# =============================================================================
#
# =============================================================================

import os
import pandas as pd
from PyEMD import CEEMDAN, EEMD
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from hyperopt import fmin, tpe, hp
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from numpy import sqrt 
import keras
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.gridspec as gridspec
import investpy as inv
import matlab.engine
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


def ApEn(U, m, r) -> float:
    """Approximate_entropy."""
    "U: seri, m: embedding dimension, r: tolerance"

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

def SampEn(L, m, r):
    """Sample entropy."""
    N = len(L)
    B = 0.0
    A = 0.0
  
    EPSILON = 1e-12
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B + EPSILON)

def calculate_entropy_StdSapma(imf,m,r):
    approx = ApEn(imf,m,r)
    sample = SampEn(imf,m,r)
    std_sapma = np.std(imf)
    return round(approx,5), round(sample,5), round(std_sapma,5)

def calculate_entropy(imfs):
    
    imf_entropy_StdDeviation = np.zeros((imfs.shape[0], 3))
    for i in range(imfs.shape[0]):
        imf_entropy_StdDeviation[i] = calculate_entropy_StdSapma(imfs[i],m,r)
    
    Entropy_StdDeviation = pd.DataFrame(imf_entropy_StdDeviation, columns = ['Approximate Entropy', 'Sample Entropy', 'Standard Deviation'])
    
    return Entropy_StdDeviation
    

def selection_imfs(imfs, Ratio_entropy):

    selected_IMFs_first = []
    high_frequency_IMFs = []
    for i in range(len(imfs)):
     
        if ((Ratio_entropy.iloc[i, 0] < 20) and (Ratio_entropy.iloc[i, 1] < 20)):
        
            selected_IMFs_first.append(imfs[i])
        else:
            high_frequency_IMFs.append(imfs[i])


    high_frequency_IMF_sum=0
    for num in high_frequency_IMFs:
        high_frequency_IMF_sum += num
    return selected_IMFs_first, high_frequency_IMFs, high_frequency_IMF_sum

def MAPE(Y_actual,Y_Predicted):
    epsilon = 1e-10

    mape = np.mean(np.abs((Y_actual - Y_Predicted)/(Y_actual+epsilon)))*100
    return mape

def repeat_list(n, x):
    return [x] * n

def relative_root_mean_squared_error(true, pred):
    n = len(true) # update
    num = np.sum(np.square(true - pred)) / n  # update
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

def rrmse(actual, predicted):

    rmse = np.sqrt(np.mean(np.square((actual - predicted)/predicted)))

    return rrmse

def smape(y_true, y_pred):

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    percentage_errors = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    return np.mean(percentage_errors) * 100


folder_path = 'E:\Hyperparameter Tunning-02092024\ICEEMDAN Denoising-Parameter Tunning\\SP500'
symbol = "^GSPC" 


start_date = "2010-01-01"
end_date = "2020-01-01"


df = yf.download(symbol, start=start_date, end=end_date).reset_index()
df = df.fillna(method='bfill')

x = df['Close'].values



eng = matlab.engine.start_matlab()
matlab_path = "E:/TİK2-ICEEMDAN-LSTM-LSTMBATCH-GRU-KOD VE SONUCLAR"
eng.addpath (matlab_path, nargout= 0 )
eng.emd_(nargout=0)

eng.iceemdan(nargout=0)
a,b = eng.iceemdan_imfs(x, nargout=2)

eng.quit()

iceemdan_imfs = pd.DataFrame(a)

scaler= StandardScaler()

train_size = int(len(df) * 0.75)
test_size = len(df) - train_size

m= 2
r=0.2


Entropy_StdDeviation_first_close = calculate_entropy(np.array(iceemdan_imfs))
Entropy_StdDeviation_first_close = Entropy_StdDeviation_first_close.fillna(method='ffill') #nan değeri varsa üstündeki değerle dolduracak
col_sum = np.sum(Entropy_StdDeviation_first_close, axis=0)
ratios = np.divide(Entropy_StdDeviation_first_close, col_sum)
Ratio_entropy = ratios*100
Ratio_entropy_first_close = Ratio_entropy.iloc[:,:2]

selected_IMFs_first_close, high_frequency_IMFs_first_close, high_frequency_IMF_sum_first_close = selection_imfs(np.array(iceemdan_imfs), Ratio_entropy_first_close)
# =============================================================================

plt.figure()

iceemdan_imfs.index = [f"IMF{i+1}" for i in range(len(iceemdan_imfs))]

colors = plt.cm.viridis(np.linspace(0, 1, len(iceemdan_imfs)+1))


fig, axes = plt.subplots(nrows=len(iceemdan_imfs)+1, ncols=1, figsize=(8, 9), sharex=True)
# Plot the original data in the first subplot
axes[0].plot(df['Close'].values)
# axs[0].plot(df)
axes[0].set_title('Original Data')

for i, (row_label, row) in enumerate(iceemdan_imfs.iterrows()):
    axes[i+1].plot(row, label=row_label, color=colors[i+1])
    #axes[i].legend()
    

    axes[i+1].text(-0.07, 0.5, row_label, rotation=90, verticalalignment='center', horizontalalignment='right', transform=axes[i+1].transAxes)


plt.xlabel('Trading Day')

plt.suptitle('First Decomposition Results', x=0.54)

plt.tight_layout()

plt.show()
# 

file_name1 = 'first_decomposition_IMFs.png'
full_path1 = os.path.join(folder_path, file_name1)
fig.savefig(full_path1)
# =============================================================================
eng_2 = matlab.engine.start_matlab()
eng_2.addpath (matlab_path, nargout= 0 )

f,g = eng_2.iceemdan_imfs(high_frequency_IMF_sum_first_close, nargout=2)
imfs_iceemdan_second = pd.DataFrame(f)
imfs_second_close = np.array (imfs_iceemdan_second)
eng_2.quit()


Entropy_StdDeviation_second_close = calculate_entropy(imfs_second_close)
col_sum_second = np.sum(Entropy_StdDeviation_second_close, axis=0)
ratios_second = np.divide(Entropy_StdDeviation_second_close, col_sum_second)
Ratio_entropy_second = ratios_second*100
Ratio_entropy_second_close = Ratio_entropy_second.iloc[:,:2]
selected_IMFs_second_close, high_frequency_IMFs_second_close, high_frequency_IMF_sum_second_close = selection_imfs(imfs_second_close, Ratio_entropy_second_close)
# =============================================================================


plt.figure()

imfs_iceemdan_second.index = [f"IMF{i+1}" for i in range(len(imfs_iceemdan_second))]


colors = plt.cm.viridis(np.linspace(0, 1, len(imfs_iceemdan_second)+1))


fig, axes = plt.subplots(nrows=len(imfs_iceemdan_second)+1, ncols=1, figsize=(8, 9), sharex=True)

axes[0].plot(high_frequency_IMF_sum_first_close)

axes[0].set_title('High Frequency Data')

for i, (row_label, row) in enumerate(imfs_iceemdan_second.iterrows()):
    axes[i+1].plot(row, label=row_label, color=colors[i+1])

    axes[i+1].text(-0.08, 0.5, row_label, rotation=90, verticalalignment='center', horizontalalignment='right', transform=axes[i+1].transAxes)


plt.xlabel('Trading Day')

plt.suptitle('Second Decomposition Results', x=0.54)

plt.tight_layout()

plt.show()
# 

file_name2 = 'second_decomposition_high_frequency_data.png'
full_path2 = os.path.join(folder_path, file_name2)
fig.savefig(full_path2)
# =============================================================================
#Ploting the IMFs selected as a result of second decomposition

plt.figure()
fig, axs = plt.subplots(nrows=len(selected_IMFs_second_close), figsize=(8, 8), sharex=True)
fig.subplots_adjust(hspace=0.2)
fig.suptitle("Selected IMFs at The 2nd Decomposition")
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_IMFs_second_close)))

# Plot the IMFs in subsequent subplots
for i in range(len(selected_IMFs_second_close)):
    axs[i].plot(selected_IMFs_second_close[i], color=colors[i])
    axs[i].set_ylabel(f'IMF {i+1}', rotation=90, ha='right', va='center')  
    axs[i].yaxis.set_label_coords(-0.1, 0.8)  

fig.text(0.5, 0.04, 'Trading Day', ha='center')

for ax in axs.flat:
    ax.set_xlabel('')
    
# Show the plot
plt.show()



file_name3 = 'selected_IMFs_second_decomp.png'
full_path3 = os.path.join(folder_path, file_name3)
fig.savefig(full_path3)
# =============================================================================


selected_all_sum_imfs = sum(selected_IMFs_first_close) + sum(selected_IMFs_second_close)
denoised_test_data_hiyerarşik = selected_all_sum_imfs[train_size:len(df)]

plt.figure()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'].values, label='Original Data')
plt.plot(selected_all_sum_imfs, label='Denoised Data')
plt.xlabel('Trading Day')
plt.legend()
plt.show()

file_name4 = 'actual_and_denoised_data_hiyerarşik_duygu.png'
full_path4 = os.path.join(folder_path, file_name4)
fig.savefig(full_path4)

# =============================================================================
# LSTM model creation
def build_lstm_model(X_train, timestep, params):
    num_layers, dropout_rate, learning_rate, batch_size = params[:4]
    layer_units = params[4:]
    
    model = Sequential()
    num_units = int(layer_units[0])
    model.add(LSTM(units=num_units, return_sequences=True if num_layers > 1 else False, input_shape=(timestep, X_train.shape[2])))
    model.add(Dropout(rate=dropout_rate))
    
    for i in range(1, num_layers - 1):
        num_units = int(layer_units[i])
        model.add(LSTM(units=num_units, return_sequences=True))
        model.add(Dropout(rate=dropout_rate))
    
    if num_layers > 1:
        num_units = int(layer_units[num_layers - 1])
        model.add(LSTM(units=num_units, return_sequences=False))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=1))
    
    return model

# GRU model creation
def build_gru_model(X_train, timestep, params):
    num_layers, dropout_rate, learning_rate, batch_size = params[:4]
    layer_units = params[4:]

    model = Sequential()
    num_units = int(layer_units[0])
    model.add(GRU(units=num_units, return_sequences=True if num_layers > 1 else False, input_shape=(timestep, X_train.shape[2])))
    model.add(Dropout(rate=dropout_rate))

    for i in range(1, num_layers - 1):
        num_units = int(layer_units[i])
        model.add(GRU(units=num_units, return_sequences=True))
        model.add(Dropout(rate=dropout_rate))

    if num_layers > 1:
        num_units = int(layer_units[num_layers - 1])
        model.add(GRU(units=num_units, return_sequences=False))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=1))

    return model

# LSTM with Batch Normalization model creation
def build_lstm_batch_model(X_train, timestep, params):
    num_layers, dropout_rate, learning_rate, batch_size = params[:4]
    layer_units = params[4:]

    model = Sequential()
    num_units = int(layer_units[0])
    model.add(LSTM(units=num_units, return_sequences=True if num_layers > 1 else False, input_shape=(timestep, X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))

    for i in range(1, num_layers - 1):
        num_units = int(layer_units[i])
        model.add(LSTM(units=num_units, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(rate=dropout_rate))

    if num_layers > 1:
        num_units = int(layer_units[num_layers - 1])
        model.add(LSTM(units=num_units, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=1))

    return model


# Objective function for hyperparameter tuning
def objective(params, model_type, X_train, y_train, X_test, y_test):
    num_layers = int(params[0])
    dropout_rate = params[1]
    learning_rate = params[2]
    batch_size = params[3]
    layer_units = params[4:]
    
    # Adjust neuron count based on number of layers
    if len(layer_units) > num_layers:
        layer_units = layer_units[:num_layers]
    elif len(layer_units) < num_layers:
        layer_units.extend([32] * (num_layers - len(layer_units)))

    if model_type == 'lstm':
        model = build_lstm_model(X_train, timestep, (num_layers, dropout_rate, learning_rate, batch_size, *layer_units))
    elif model_type == 'gru':
        model = build_gru_model(X_train, timestep, (num_layers, dropout_rate, learning_rate, batch_size, *layer_units))
    elif model_type == 'lstm_batch':
        model = build_lstm_batch_model(X_train, timestep, (num_layers, dropout_rate, learning_rate, batch_size, *layer_units))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=10, batch_size=int(batch_size), verbose=0)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return mse

# Define the search space
max_layers = 6  # Maximum number of layers
search_space = [
    Integer(1, max_layers, name='num_layers'),
    Real(0.001, 0.5, name='dropout_rate'),
    Real(1e-5, 1e-2, name='learning_rate'),
    Integer(8, 256, name='batch_size'),
] + [Integer(8, 1024, name=f'layer_units_{i}') for i in range(max_layers)]



def create_dataset(data, timestep=1):
    X, y = [], []
    for i in range(len(data) - timestep):
        X.append(data[i:(i + timestep), 0])  
        y.append(data[i + timestep, 0])     
    return np.array(X), np.array(y)


# IMF data processing and model training
def process_imfs(imfs, level_name, timestep):
    train_results = []
    test_results = []
    best_params_all_imfs = []
    y_test_values = []
    y_pred_values = []

    for idx, imf in enumerate(imfs):
        print(f"Optimizing for IMF {idx + 1} ({level_name})")

        # Reshape and scale the IMF data
        X = scaler.fit_transform(imf.reshape(-1, 1))

        # Windowing işlemi ile tüm veriyi X ve y setlerine oluştur
        X_all, y_all = create_dataset(X, timestep)

        # Eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)

        # X_train ve X_test'i 3D forma dönüştür
        X_train = X_train.reshape((X_train.shape[0], timestep, 1))
        X_test = X_test.reshape((X_test.shape[0], timestep, 1))

        # Bayesian optimization for LSTM, GRU, and LSTM-Batch
        result_lstm = gp_minimize(lambda x: objective(x, 'lstm', X_train, y_train, X_test, y_test), search_space, n_calls=25, random_state=0)
        result_gru = gp_minimize(lambda x: objective(x, 'gru', X_train, y_train, X_test, y_test), search_space, n_calls=25, random_state=0)
        result_lstm_batch = gp_minimize(lambda x: objective(x, 'lstm_batch', X_train, y_train, X_test, y_test), search_space, n_calls=25, random_state=0)

       
        best_params_lstm = result_lstm.x
        best_params_gru = result_gru.x
        best_params_lstm_batch = result_lstm_batch.x

      
        best_params = {
            'IMF': idx + 1,
            'LSTM': best_params_lstm,
            'GRU': best_params_gru,
            'LSTM-Batch': best_params_lstm_batch
        }

       
        best_params_all_imfs.append(best_params)

        # En iyi hiperparametrelerle modelleri oluştur ve eğit
        print(f"Training LSTM model for IMF {idx + 1} ({level_name})...")
        lstm_model = build_lstm_model(X_train, timestep, best_params_lstm)
        lstm_model.compile(optimizer=Adam(learning_rate=best_params_lstm[2]), loss='mean_squared_error')
        lstm_model.fit(X_train, y_train, epochs=100, batch_size=int(best_params_lstm[3]), verbose=0)
        lstm_train_pred = lstm_model.predict(X_train)
        lstm_test_pred = lstm_model.predict(X_test)

        print(f"Training GRU model for IMF {idx + 1} ({level_name})...")
        gru_model = build_gru_model(X_train, timestep, best_params_gru)
        gru_model.compile(optimizer=Adam(learning_rate=best_params_gru[2]), loss='mean_squared_error')
        gru_model.fit(X_train, y_train, epochs=100, batch_size=int(best_params_gru[3]), verbose=0)
        gru_train_pred = gru_model.predict(X_train)
        gru_test_pred = gru_model.predict(X_test)

        print(f"Training LSTM-Batch Normalization model for IMF {idx + 1} ({level_name})...")
        lstm_batch_model = build_lstm_batch_model(X_train, timestep, best_params_lstm_batch)
        lstm_batch_model.compile(optimizer=Adam(learning_rate=best_params_lstm_batch[2]), loss='mean_squared_error')
        lstm_batch_model.fit(X_train, y_train, epochs=100, batch_size=int(best_params_lstm_batch[3]), verbose=0)
        lstm_batch_train_pred = lstm_batch_model.predict(X_train)
        lstm_batch_test_pred = lstm_batch_model.predict(X_test)

        # Eğitim ve test sonuçlarını kaydet (MSE, MAE, MAPE, R²)
        train_results.append({
            'IMF': idx + 1,
            'LSTM': {
                'MSE': mean_squared_error(y_train, lstm_train_pred),
                'MAE': mean_absolute_error(y_train, lstm_train_pred),
                'MAPE': mean_absolute_percentage_error(y_train, lstm_train_pred),
                'R2': r2_score(y_train, lstm_train_pred)
            },
            'GRU': {
                'MSE': mean_squared_error(y_train, gru_train_pred),
                'MAE': mean_absolute_error(y_train, gru_train_pred),
                'MAPE': mean_absolute_percentage_error(y_train, gru_train_pred),
                'R2': r2_score(y_train, gru_train_pred)
            },
            'LSTM-Batch': {
                'MSE': mean_squared_error(y_train, lstm_batch_train_pred),
                'MAE': mean_absolute_error(y_train, lstm_batch_train_pred),
                'MAPE': mean_absolute_percentage_error(y_train, lstm_batch_train_pred),
                'R2': r2_score(y_train, lstm_batch_train_pred)
            }
        })

        test_results.append({
            'IMF': idx + 1,
            'LSTM': {
                'MSE': mean_squared_error(y_test, lstm_test_pred),
                'MAE': mean_absolute_error(y_test, lstm_test_pred),
                'MAPE': mean_absolute_percentage_error(y_test, lstm_test_pred),
                'R2': r2_score(y_test, lstm_test_pred)
            },
            'GRU': {
                'MSE': mean_squared_error(y_test, gru_test_pred),
                'MAE': mean_absolute_error(y_test, gru_test_pred),
                'MAPE': mean_absolute_percentage_error(y_test, gru_test_pred),
                'R2': r2_score(y_test, gru_test_pred)
            },
            'LSTM-Batch': {
                'MSE': mean_squared_error(y_test, lstm_batch_test_pred),
                'MAE': mean_absolute_error(y_test, lstm_batch_test_pred),
                'MAPE': mean_absolute_percentage_error(y_test, lstm_batch_test_pred),
                'R2': r2_score(y_test, lstm_batch_test_pred)
            }
        })
        
      
        y_test_values.append({
            'IMF': idx + 1,
            'y_test': y_test
        })
        
        y_pred_values.append({
            'IMF': idx + 1,
            'LSTM': lstm_test_pred,
            'GRU': gru_test_pred,
            'LSTM-Batch': lstm_batch_test_pred
        })
    
    return train_results, test_results, best_params_all_imfs, y_test_values, y_pred_values

timestep = 10
train_results_first_level, test_results_first_level, best_params_first_level,y_test_values_first_level, y_pred_values_first_level = process_imfs(selected_IMFs_first_close, "First Level", timestep)
train_results_second_level, test_results_second_level, best_params_second_level, y_test_values_second_level, y_pred_values_second_level = process_imfs(selected_IMFs_second_close, "Second Level", timestep)




def convert_results_to_dataframe(results):
    data = []
    
    
    for imf_result in results:
        imf_number = imf_result['IMF']
        
        # LSTM sonuçlarını ekle
        data.append({
            'IMF': imf_number,
            'Model': 'LSTM',
            'MSE': imf_result['LSTM']['MSE'],
            'MAE': imf_result['LSTM']['MAE'],
            'MAPE': imf_result['LSTM']['MAPE'],
            'R2': imf_result['LSTM']['R2']
        })
        
        # GRU 
        data.append({
            'IMF': imf_number,
            'Model': 'GRU',
            'MSE': imf_result['GRU']['MSE'],
            'MAE': imf_result['GRU']['MAE'],
            'MAPE': imf_result['GRU']['MAPE'],
            'R2': imf_result['GRU']['R2']
        })
        
        # LSTM-Batch 
        data.append({
            'IMF': imf_number,
            'Model': 'LSTM-Batch',
            'MSE': imf_result['LSTM-Batch']['MSE'],
            'MAE': imf_result['LSTM-Batch']['MAE'],
            'MAPE': imf_result['LSTM-Batch']['MAPE'],
            'R2': imf_result['LSTM-Batch']['R2']
        })
    
    return pd.DataFrame(data)


train_results_df_first = convert_results_to_dataframe(train_results_first_level)
test_results_df_first = convert_results_to_dataframe(test_results_first_level)
best_params_df_first = pd.DataFrame(best_params_first_level)

train_results_df_second = convert_results_to_dataframe(train_results_second_level)
test_results_df_second = convert_results_to_dataframe(test_results_second_level)
best_params_df_second = pd.DataFrame(best_params_second_level)


    

def get_best_model_predictions(test_results, y_test_values, y_pred_values):
    best_model_predictions = []
    

    for idx, imf_result in enumerate(test_results):
        imf_number = imf_result['IMF']
        

        min_mse = min(imf_result['LSTM']['MSE'], imf_result['GRU']['MSE'], imf_result['LSTM-Batch']['MSE'])
        

        if min_mse == imf_result['LSTM']['MSE']:
            best_model = 'LSTM'
            best_y_pred = y_pred_values[idx]['LSTM']
        elif min_mse == imf_result['GRU']['MSE']:
            best_model = 'GRU'
            best_y_pred = y_pred_values[idx]['GRU']
        else:
            best_model = 'LSTM-Batch'
            best_y_pred = y_pred_values[idx]['LSTM-Batch']
        

        best_model_predictions.append({
            'IMF': imf_number,
            'Best Model': best_model,
            'MSE': min_mse,
            'MAE': imf_result[best_model]['MAE'],
            'MAPE': imf_result[best_model]['MAPE'],
            'R2': imf_result[best_model]['R2'],
            'y_test': y_test_values[idx]['y_test'],
            'y_pred': best_y_pred 
        })
    
    return pd.DataFrame(best_model_predictions)



best_model_predictions_df_first = get_best_model_predictions(test_results_first_level, y_test_values_first_level, y_pred_values_first_level)
best_model_predictions_df_second = get_best_model_predictions(test_results_second_level, y_test_values_second_level, y_pred_values_second_level)


symbol_cleaned = symbol.replace("^", "").replace("/", "_").replace("\\", "_")


file_name = f"{symbol_cleaned}_results.xlsx"


file_path = os.path.join(folder_path, file_name)


try:
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # Verileri yazma
        train_results_df_first.to_excel(writer, sheet_name='Train Results First Level', index=False)
        test_results_df_first.to_excel(writer, sheet_name='Test Results First Level', index=False)
        best_params_df_first.to_excel(writer, sheet_name='Best Params First Level', index=False)
        best_model_predictions_df_first.to_excel(writer, sheet_name='Best Model Predictions First Level', index=False)

        train_results_df_second.to_excel(writer, sheet_name='Train Results Second Level', index=False)
        test_results_df_second.to_excel(writer, sheet_name='Test Results Second Level', index=False)
        best_params_df_second.to_excel(writer, sheet_name='Best Params Second Level', index=False)
        best_model_predictions_df_second.to_excel(writer, sheet_name='Best Model Predictions Second Level', index=False)

      
        workbook  = writer.book
        number_format = workbook.add_format({'num_format': '0.000000'}) 
        
        for sheet_name in ['Train Results First Level', 'Test Results First Level', 'Best Params First Level',
                           'Train Results Second Level', 'Test Results Second Level', 'Best Params Second Level']:
            worksheet = workbook.get_worksheet_by_name(sheet_name)
            for col_num in range(len(writer.sheets[sheet_name].columns)):
                worksheet.set_column(col_num, col_num, None, number_format)

    print(f"Veriler başarıyla şu dosyaya yazıldı: {file_path}")


except Exception as e:
    print(f"Bir hata oluştu: {e}")

    print(f"Bir hata oluştu: {e}")
    



