# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:06:13 2024

@author: Spectre
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yfinance as yf
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
import math
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR



scaler= StandardScaler()
timestep = 10
folder_path = 'E:\decomposition-parameter optimization\\MIGROS'
# symbol = "AKBNK.IS" #akbank hissesi
# symbol = "KCHOL.IS" #KCHOL hissesi
# symbol = "THYAO.IS" #THYAO hissesi
# symbol= "ULKER.IS"#ULKER hissesi
symbol= "MGROS.IS"#MIGROS hissesi


# Model oluşturma fonksiyonları
def build_lstm_model(X_train, timestep, params):
    num_layers, dropout_rate, learning_rate, batch_size = params[:4]
    layer_units = params[4:]
    
    model = Sequential()
    num_units = int(layer_units[0])
    model.add(LSTM(units=num_units, return_sequences=True if num_layers > 1 else False, input_shape=(timestep, X_train.shape[2])))
 
    
    if num_layers > 1:
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


def build_gru_model(X_train, timestep, params):
    num_layers, dropout_rate, learning_rate, batch_size = params[:4]
    layer_units = params[4:]

    model = Sequential()
    num_units = int(layer_units[0])
    model.add(GRU(units=num_units, return_sequences=True if num_layers > 1 else False, input_shape=(timestep, X_train.shape[2])))
    
    if num_layers > 1:
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

def build_lstm_batch_model(X_train, timestep, params):
    num_layers, dropout_rate, learning_rate, batch_size = params[:4]
    layer_units = params[4:]

    model = Sequential()
    num_units = int(layer_units[0])
    model.add(LSTM(units=num_units, return_sequences=True if num_layers > 1 else False, input_shape=(timestep, X_train.shape[2])))
    model.add(BatchNormalization())
    
    if num_layers > 1:
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


def process_and_predict(level_name):
    # Best parameters and training/test data file
    
     
    filename = os.path.join(folder_path, f'best_params_and_data_{level_name}.pkl')
    data = joblib.load(filename)
    
    best_params_all_imfs = data['best_params']
    training_data_all_imfs = data['training_data']
    
    train_results = []
    test_results = []
    y_test_values = []
    y_pred_values = []


    for best_params, training_data in zip(best_params_all_imfs, training_data_all_imfs):
        imf = best_params['IMF']
        print(f"Processing IMF {imf} ({level_name})")
        
        X_train = np.array(training_data['X_train'])
        y_train = np.array(training_data['y_train'])
        X_test = np.array(training_data['X_test'])
        y_test = np.array(training_data['y_test'])
        
     
        scaler_filename = os.path.join(folder_path, f'scaler_IMF_{imf}_{level_name}.pkl')
        scaler = joblib.load(scaler_filename)
        
        # LSTM model
        print("LSTM Training")
        lstm_params = best_params['LSTM']
        lstm_model = build_lstm_model(X_train, timestep, lstm_params)
        lstm_model.compile(optimizer=Adam(learning_rate=lstm_params[2]), loss='mean_squared_error')
        lstm_model.fit(X_train, y_train, epochs=100, batch_size=int(lstm_params[3]), verbose=0)
        lstm_train_pred = lstm_model.predict(X_train)
        lstm_test_pred = lstm_model.predict(X_test)
        
       
        lstm_test_pred_rescaled = scaler.inverse_transform(lstm_test_pred.reshape(-1, 1)).flatten()
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        

        # GRU model
        print("GRU Training")
        gru_params = best_params['GRU']
        gru_model = build_gru_model(X_train, timestep, gru_params)
        gru_model.compile(optimizer=Adam(learning_rate=gru_params[2]), loss='mean_squared_error')
        gru_model.fit(X_train, y_train, epochs=100, batch_size=int(gru_params[3]), verbose=0)
        gru_train_pred = gru_model.predict(X_train)
        gru_test_pred = gru_model.predict(X_test)
        
      
        gru_test_pred_rescaled = scaler.inverse_transform(gru_test_pred.reshape(-1, 1)).flatten()


        # LSTM-Batch model
        print("LSTM-Batch Training")
        lstm_batch_params = best_params['LSTM-Batch']
        lstm_batch_model = build_lstm_batch_model(X_train, timestep, lstm_batch_params)
        lstm_batch_model.compile(optimizer=Adam(learning_rate=lstm_batch_params[2]), loss='mean_squared_error')
        lstm_batch_model.fit(X_train, y_train, epochs=100, batch_size=int(lstm_batch_params[3]), verbose=0)
        lstm_batch_train_pred = lstm_batch_model.predict(X_train)
        lstm_batch_test_pred = lstm_batch_model.predict(X_test)
        
       
        lstm_batch_test_pred_rescaled = scaler.inverse_transform(lstm_batch_test_pred.reshape(-1, 1)).flatten()



        # Training and testing results
        train_results.append({
            'IMF': imf,
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
            'IMF': imf,
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
            'IMF': imf,
            'y_test_scaled': y_test, 
            'y_test_rescaled': y_test_rescaled  
        })
        
        y_pred_values.append({
            'IMF': imf,
            'LSTM': lstm_test_pred, 
            'GRU': gru_test_pred, 
            'LSTM-Batch': lstm_batch_test_pred, 
            'LSTM_rescaled': lstm_test_pred_rescaled,
            'GRU_rescaled': gru_test_pred_rescaled,  
            'LSTM-Batch_rescaled': lstm_batch_test_pred_rescaled  
        })
        

    return train_results, test_results, y_test_values, y_pred_values


train_results_first_level, test_results_first_level, y_test_values_first_level, y_pred_values_first_level = process_and_predict("First Level")
train_results_second_level, test_results_second_level, y_test_values_second_level, y_pred_values_second_level = process_and_predict("Second Level")


def convert_results_to_dataframe(results):
    data = []
    

    for imf_result in results:
        imf_number = imf_result['IMF']
        
       
        data.append({
            'IMF': imf_number,
            'Model': 'LSTM',
            'MSE': imf_result['LSTM']['MSE'],
            'MAE': imf_result['LSTM']['MAE'],
            'MAPE': imf_result['LSTM']['MAPE'],
            'R2': imf_result['LSTM']['R2']
        })
        
    
        data.append({
            'IMF': imf_number,
            'Model': 'GRU',
            'MSE': imf_result['GRU']['MSE'],
            'MAE': imf_result['GRU']['MAE'],
            'MAPE': imf_result['GRU']['MAPE'],
            'R2': imf_result['GRU']['R2']
        })
       
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


train_results_df_second = convert_results_to_dataframe(train_results_second_level)
test_results_df_second = convert_results_to_dataframe(test_results_second_level)



def get_best_model_predictions(test_results, y_test_values, y_pred_values):
    best_model_predictions = []
    
 
    for idx, imf_result in enumerate(test_results):
        imf_number = imf_result['IMF']
        
        
        min_mse = min(imf_result['LSTM']['MSE'], imf_result['GRU']['MSE'], imf_result['LSTM-Batch']['MSE'])
        
       
        if min_mse == imf_result['LSTM']['MSE']:
            best_model = 'LSTM'
            best_y_pred_scaled = y_pred_values[idx]['LSTM']
            best_y_pred_rescaled = y_pred_values[idx]['LSTM_rescaled']
            
        elif min_mse == imf_result['GRU']['MSE']:
            best_model = 'GRU'
            best_y_pred_scaled = y_pred_values[idx]['GRU']
            best_y_pred_rescaled = y_pred_values[idx]['GRU_rescaled']
            
        else:
            best_model = 'LSTM-Batch'
            best_y_pred_scaled = y_pred_values[idx]['LSTM-Batch']
            best_y_pred_rescaled = y_pred_values[idx]['LSTM-Batch_rescaled']
            
        best_model_predictions.append({
            'IMF': imf_number,
            'Best Model': best_model,
            'MSE': min_mse,
            'MAE': imf_result[best_model]['MAE'],
            'MAPE': imf_result[best_model]['MAPE'],
            'R2': imf_result[best_model]['R2'],
            'y_test_scaled': np.array(y_test_values[idx]['y_test_scaled']).flatten(), 
            'y_test_rescaled': np.array(y_test_values[idx]['y_test_rescaled']).flatten(), 
            'y_pred_scaled': np.array(best_y_pred_scaled).flatten(),  
            'y_pred_rescaled': np.array(best_y_pred_rescaled).flatten()  
        })
    
    return pd.DataFrame(best_model_predictions)


best_model_predictions_df_first = get_best_model_predictions(test_results_first_level, y_test_values_first_level, y_pred_values_first_level )
best_model_predictions_df_second = get_best_model_predictions(test_results_second_level, y_test_values_second_level, y_pred_values_second_level)


symbol_cleaned = symbol.replace("^", "").replace("/", "_").replace("\\", "_")


file_name = f"{symbol_cleaned}_results.xlsx"

file_path = os.path.join(folder_path, file_name)


try:
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
      
        train_results_df_first.to_excel(writer, sheet_name='Train Results 1st', index=False)
        test_results_df_first.to_excel(writer, sheet_name='Test Results 1st', index=False)
        best_model_predictions_df_first.to_excel(writer, sheet_name='Best Predictions 1st', index=False)

        train_results_df_second.to_excel(writer, sheet_name='Train Results 2nd', index=False)
        test_results_df_second.to_excel(writer, sheet_name='Test Results 2nd', index=False)
        best_model_predictions_df_second.to_excel(writer, sheet_name='Best Predictions 2nd', index=False)

        
        workbook = writer.book
        number_format = workbook.add_format({'num_format': '0.000000'}) 
        
        for sheet_name, dataframe in zip(
            ['Train Results 1st', 'Test Results 1st', 'Best Predictions 1st',
             'Train Results 2nd', 'Test Results 2nd', 'Best Predictions 2nd'],
            [train_results_df_first, test_results_df_first, best_model_predictions_df_first,
             train_results_df_second, test_results_df_second, best_model_predictions_df_second]
        ):
            worksheet = workbook.get_worksheet_by_name(sheet_name)
            for col_num in range(dataframe.shape[1]):  
                worksheet.set_column(col_num, col_num, None, number_format)

    print(f"Veriler başarıyla şu dosyaya yazıldı: {file_path}")

except Exception as e:
    print(f"Bir hata oluştu: {e}")


y_test_first_sum_rescaled = best_model_predictions_df_first['y_test_rescaled'].sum()
y_test_second_sum_rescaled = best_model_predictions_df_second['y_test_rescaled'].sum()


final_y_test_rescaled = y_test_first_sum_rescaled + y_test_second_sum_rescaled


y_pred_first_sum_rescaled = best_model_predictions_df_first['y_pred_rescaled'].sum()
y_pred_second_sum_rescaled = best_model_predictions_df_second['y_pred_rescaled'].sum()


final_y_pred_rescaled = y_pred_first_sum_rescaled + y_pred_second_sum_rescaled

# =============================================================================
# 

y_test_first_sum_scaled = best_model_predictions_df_first['y_test_scaled'].sum()
y_test_second_sum_scaled = best_model_predictions_df_second['y_test_scaled'].sum()


final_y_test_scaled = y_test_first_sum_scaled + y_test_second_sum_scaled

y_pred_first_sum_scaled = best_model_predictions_df_first['y_pred_scaled'].sum()
y_pred_second_sum_scaled = best_model_predictions_df_second['y_pred_scaled'].sum()


final_y_pred_scaled = y_pred_first_sum_scaled + y_pred_second_sum_scaled


# =============================================================================
# SCALED DATA

plt.figure()
fig = plt.figure(figsize=(12, 6))
plt.plot(final_y_test_scaled, label='Denoised time serie')
plt.plot(final_y_pred_scaled, label='Predicted time serie')
#plt.plot(denoised_test_data_hiyerarşik, label='Denoised Value')

plt.xlabel('Trading Day')

plt.legend()
plt.show()

file_name_final = 'predicted_and_original_method_final_result_graphics_scaled_data.png'

full_path5 = os.path.join(folder_path, file_name_final)
fig.savefig(full_path5)
# =============================================================================
# RESCALED DATA

plt.figure()
fig = plt.figure(figsize=(12, 6))
plt.plot(final_y_test_rescaled, label='Denoised time serie')
plt.plot(final_y_pred_rescaled, label='Predicted time serie')

plt.xlabel('Trading Day')

plt.legend()
plt.show()

file_name_final = 'predicted_and_original_method_final_result_graphics_rescaled_data.png'

full_path6 = os.path.join(folder_path, file_name_final)
fig.savefig(full_path6)
# =========================================================================
final_result_path = os.path.join(folder_path, "final_results_test_and_predict.xlsx")
final_results_df = pd.DataFrame({
    'final_y_test_rescaled': final_y_test_rescaled.flatten(),
    'final_y_pred_rescaled': final_y_pred_rescaled.flatten(),
    'final_y_test_scaled': final_y_test_scaled.flatten(),
    'final_y_pred_scaled': final_y_pred_scaled.flatten()
})

# Excel dosyasına kaydet
final_results_df.to_excel(final_result_path, index=False)

print(f"Veriler '{final_result_path}' dosyasına başarıyla kaydedildi.")
# =============================================================================




from scipy.stats import linregress
import os


def piecewise_linear_representation(data, segment_length):
    n_points = len(data)
    segments = []

    for start in range(0, n_points, segment_length):
        end = min(start + segment_length, n_points)
        segment_data = data[start:end]
        x = np.arange(len(segment_data))
        slope, intercept, r_value, p_value, std_err = linregress(x, segment_data)
        segments.append((start, end, slope, intercept))
    
    return segments

def generate_signals(segments):
    signals = []
    last_signal = None  # Son sinyali takip etmek için
    for i in range(1, len(segments)):
        prev_slope = segments[i-1][2]
        curr_slope = segments[i][2]
        
        if prev_slope < 0 and curr_slope > 0 and (last_signal is None or last_signal == 'SELL'):
            
            signals.append((segments[i][0], 'BUY'))
            last_signal = 'BUY'
        elif prev_slope > 0 and curr_slope < 0 and last_signal == 'BUY':
           
            signals.append((segments[i][0], 'SELL'))
            last_signal = 'SELL'
    
    return signals
def plot_data_with_signals(data, segments, signals):
    plt.figure(figsize=(14, 7))
    plt.plot(data, label='Close Price', color='blue', alpha=0.5)
    
    # Use different colors for each segment, but without labels for the legend
    colors = plt.cm.get_cmap('tab10', len(segments))
    
    for idx, (start, end, slope, intercept) in enumerate(segments):
        x = np.arange(start, end)
        plt.plot(data.index[start:end], slope * (x - start) + intercept, color=colors(idx))
    
    # Plot signals (BUY/SELL) with labels for the legend
    for (index, signal) in signals:
        plt.scatter(data.index[index], data[index], color='red' if signal == 'SELL' else 'green', marker='o', s=100, label=f'{signal}')
    
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

   
    plt.xlabel('Trading Day')
    plt.ylabel('Close Price')
    plt.show()
    
def plot_data_with_signals_TR(data, segments, signals):
    plt.figure(figsize=(14, 7))
    plt.plot(data, label='Kapanış Fiyatı', color='blue', alpha=0.5)
    
    
    colors = plt.cm.get_cmap('tab10', len(segments))
    
    for idx, (start, end, slope, intercept) in enumerate(segments):
        x = np.arange(start, end)
        plt.plot(data.index[start:end], slope * (x - start) + intercept, color=colors(idx))
    
    # Al/Sat sinyallerini çizme (kırmızı: SAT, yeşil: AL)
    for (index, signal) in signals:
        color = 'red' if signal == 'SELL' else 'green'
        label = 'SAT' if signal == 'SELL' else 'AL'
        plt.scatter(data.index[index], data[index], color=color, marker='o', s=100, label=label)
    
    # Tekrarlanan etiketleri kaldırma
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.xlabel('İşlem Günü')
    plt.ylabel('Kapanış Fiyatı')
    plt.show()



def financial_evaluation(data, signals, initial_capital=10000):
    capital = initial_capital
    position = 0  
    
    for i in range(len(signals)):
        index, signal = signals[i]
        price = data.iloc[index]  
        
        if signal == 'BUY' and position == 0:  
            position = capital // price  
            capital-= ((position*price) + (position*price*0.001)) 
        elif signal == 'SELL' and position > 0:  
            capital += position * price   
            capital-= (position*price*0.0001)
            position = 0  
    
   
    if position > 0:
        capital += position * data.iloc[-1]  
        capital-= (position*price*0.001)
        position = 0  
    
    return capital - initial_capital  



test_data = final_y_pred_rescaled


segment_length = 10
initial_capital = 10000

# Piecewise Linear Representation

segments = piecewise_linear_representation(test_data, segment_length)

# Generate Signals
signals = generate_signals(segments)

# Plot data with signals (test verisi üzerinde)
plot_data_with_signals(test_data, segments, signals)
graph_name = 'Financial_evaluation_PLR_ENG.png'
full_path = os.path.join(folder_path, graph_name)
plt.savefig(full_path)





profit_loss = financial_evaluation(test_data, signals, initial_capital)
print(f"Total Profit/Loss: {profit_loss:.2f} USD")