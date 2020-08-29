import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras.backend as K
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from sklearn import metrics

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
""" 
parse_dates is used by Pandas to automatically recognize dates.
Pandas implicitly recognizes the format by agr infer_datetime_format=True
https://stackoverflow.com/questions/17465045/can-pandas-automatically-recognize-dates
"""

df = pd.read_csv('../csv_files/main_dataset.csv', parse_dates=['timestamp'], index_col="timestamp",
                 infer_datetime_format=True)
print(df.head())
print(df.describe())

# 90 percent of the data is used for training
train_size = int(len(df) * 0.9)
# rest of the data is used for test
test_size = len(df) - train_size
# split actual dataset into test and train variables
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))
# (1300, 14) (145, 14), 1300 examples for train and 145 examples for test
print(train.shape, test.shape)

# For RF_BiLSTM
f_columns = ['max_demand_gen', 'highest_gen', 'min_gen', 'day_peak_gen', 'eve_peak_gen']

# For biLSTM
# f_columns = ['max_demand_gen', 'highest_gen', 'min_gen', 'day_peak_gen', 'eve_peak_gen', 'eve_peak_load_shedding',
#              'max_temp', 'total_gas', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'season']

f_transformer = RobustScaler()
total_energy_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
total_energy_transformer = total_energy_transformer.fit(train[['total_energy']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['total_energy'] = total_energy_transformer.transform(train[['total_energy']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['total_energy'] = total_energy_transformer.transform(test[['total_energy']])

# get sequences of data

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# take last 10 days to predict the data of the next day
time_steps = 10

# reshape to [samples, time_steps, n_features]
# (1013, 10, 14) (1013,)
X_train, y_train = create_dataset(train, train.total_energy, time_steps)
X_test, y_test = create_dataset(test, test.total_energy, time_steps)

# reshape to [samples, time_steps, n_features]
# (1013, 10, 14) (1013,)
print(X_train.shape, y_train.shape)


def percentage_difference(y_true, y_pred):
    return K.mean(abs(y_pred/y_true - 1) * 100)

modelRFBiLstm = keras.Sequential()
modelRFBiLstm.add(
    keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=128,  # units is number of neurons
                input_shape=(X_train.shape[1], X_train.shape[2])  # 10, 14
            )
        )
)
modelRFBiLstm.add(keras.layers.Dropout(rate=0.2))
modelRFBiLstm.add(keras.layers.Dense(units=1))
modelRFBiLstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', percentage_difference])

historyRFBiLstm = modelRFBiLstm.fit(
    X_train, y_train,
    epochs=700,
    batch_size=512,
    validation_split=0.1,
    shuffle=False
)


plt.clf()
loss = historyRFBiLstm.history['loss']
print("training loss")
print(loss)
val_loss = historyRFBiLstm.history['val_loss']
print("validation loss")
print(val_loss)
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='training loss', linewidth=3)
plt.plot(epochs, val_loss, 'y', label='validation loss', linewidth=3)
plt.title('training and validation loss')
plt.xlabel('epochs', fontsize=30)
plt.ylabel('loss', fontsize=30)
plt.legend(fontsize=25)
plt.savefig('../asset/loss_graph.png', bbox_inches='tight')
plt.show()

y_pred_rf_bi_lstm = modelRFBiLstm.predict(X_test)
y_train_inv = total_energy_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = total_energy_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv_rf_bi_lstm = total_energy_transformer.inverse_transform(y_pred_rf_bi_lstm)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MSE rf-bi-LSTM: ', metrics.mean_squared_error(y_test, y_pred_rf_bi_lstm))
print('RMSE rf-bi-LSTM: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf_bi_lstm)))
print('MAE rf-bi-LSTM: ', metrics.mean_absolute_error(y_test, y_pred_rf_bi_lstm))
print('MAPE rf-bi-LSTM: ', mean_absolute_percentage_error(y_test, y_pred_rf_bi_lstm))

plt.clf()
plt.plot(y_test_inv.flatten(), 'b', marker='.', label="true value", linewidth=3, markersize='12')
plt.plot(y_pred_inv_rf_bi_lstm.flatten(), 'r', marker='.', label="rf-bi-LSTM", linewidth=3, markersize='12')
plt.ylabel('electricity Consumption (MKWh)', fontsize=30)
plt.xlabel('time step', fontsize=30)
plt.legend(fontsize=25)
plt.savefig('../assets/prediction_result_RF_biLSTM.png', bbox_inches='tight')
# plt.savefig('../assets_lstm_2/total_test_vs_train_lstm_imp_feat.png', bbox_inches='tight')
# plt.savefig('../test_asset/total_test_vs_train_imp_feat.png', bbox_inches='tight')
plt.show()


