import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import yfinance as yf
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = load_model('model.stockmarket')
start = dt.datetime(2016,1,1) 
end = dt.datetime(2021,12,1) 
df = yf.Ticker('USDIDR=X') 
data = df.history(start=start, end=end)

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60
test_start=dt.datetime(2021,12,1)
test_end = dt.datetime.now()
 
test_data = df.history(start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset=pd.concat((data['Close'], test_data['Close']),axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)- prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color = 'blue', label = "Actual  Price")
plt.plot(predicted_prices, color='green', label="Predicted Prices")
plt.title(" Share Price")
plt.xlabel('time')
plt.ylabel('Share Price')
plt.legend()
plt.show()

real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")