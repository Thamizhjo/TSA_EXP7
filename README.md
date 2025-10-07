# AUTO-REGRESSIVE-MODEL


### AIM:
To Implementat an Auto Regressive Model using Python in Gold Price Prediction.
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('results.csv')

data['date'] = pd.to_datetime(data['date'])
data['home_score'] = pd.to_numeric(data['home_score'], errors='coerce')
data = data.dropna(subset=['home_score'])

data.set_index('date', inplace=True)

result = adfuller(data['home_score'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

train_size = int(len(data) * 0.8)
train_data, test_data = data[0:train_size], data[train_size:]

model = AutoReg(train_data['home_score'], lags=13)
ar_model_fit = model.fit()

plt.figure(figsize=(10, 5))
plot_acf(train_data['home_score'], lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(train_data['home_score'], lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = ar_model_fit.predict(
    start=len(train_data),
    end=len(train_data) + len(test_data) - 1,
    dynamic=False
)

plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['home_score'], label='Actual Test Data', color='blue')
plt.plot(test_data.index, predictions, label='Predicted Data', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Home Scores')
plt.xlabel('Date')
plt.ylabel('Home Score')
plt.legend()
plt.grid(True)
plt.show()

mse = mean_squared_error(test_data['home_score'], predictions)
print('Mean Squared Error (MSE):', mse)
```

### OUTPUT:

## GIVEN DATA:

## AUGMENTED DICKEY-FULLER TEST:

<img width="404" height="48" alt="image" src="https://github.com/user-attachments/assets/75e11b1e-4f41-4901-a470-78ba08330f26" />

## PACF - ACF:
<img width="720" height="531" alt="image" src="https://github.com/user-attachments/assets/3591afae-03c0-4a1b-a7b8-68420f2e8988" />

<img width="727" height="531" alt="image" src="https://github.com/user-attachments/assets/0f065a90-26af-4101-b8ee-59fd7434ecca" />


## PREDICTION:

## FINIAL PREDICTION:

<img width="1319" height="667" alt="image" src="https://github.com/user-attachments/assets/7de2b82d-686a-42fd-9919-43f93477c509" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
