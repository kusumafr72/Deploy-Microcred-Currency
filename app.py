import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from flask import flash, render_template, request, send_from_directory

app=flash(__name__)
model = load_model('model.stockmarket')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    currency = flash(str(request.form['currency']))
    df = yf.Ticker("{currency}")
    data = df.history()


    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    prediction_days = 60
    test_start=dt.datetime(2021,1,1)
    test_end = dt.datetime.now()


 
    test_data = df.history(start=test_start, end=test_end)
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'] # one or more

    Q1 = data[cols].quantile(0.25)
    Q3 = data[cols].quantile(0.75)
    IQR = Q3 - Q1

    test_data = test_data[~((test_data[cols] < (Q1 - 3 * IQR)) |(test_data[cols] > (Q3 + 3 * IQR))).any(axis=1)]

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

    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    output = prediction
    return render_template('index.html', prediction_text='Tomorrow Prices Around Rp {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
