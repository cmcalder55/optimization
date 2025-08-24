
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
# import yfinance as yf

def plotPrice(df,n):
    plt.figure()
    title = 'Tesla Stock Price for the Past '+str(n)+' Days'
    df.plot(x='Date', y = 'Stock Price', title=title, legend=False)
    plt.xticks(rotation=70)
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.show()

 
def formatData(filepath,n):
    df = pd.read_csv(filepath, header=0)
    data = {'Date':df['Date'], 'Stock Price':df['Close']}
    df = pd.DataFrame(data)

    dates = pd.to_datetime(df['Date'])
    df['Date'] = (dates-dates.min())/np.timedelta64(1,'D')
    return df, dates


def SGDregression(df):
        X = df['Date'].values.reshape(-1, 1)
        y = df['Stock Price'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

        # Fit scaler on training data only
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        X_train_scaled = scaler_X.transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

        # Train model
        sgdr = SGDRegressor()
        sgdr.fit(X_train_scaled, y_train_scaled)

        # Predict
        y_pred_scaled = sgdr.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # Build DataFrame for plotting
        test_idx = np.argsort(X_test.flatten())
        dd = pd.DataFrame({
            'Day': X_test.flatten()[test_idx],
            'Predicted Price': y_pred[test_idx],
            'Actual Price': y_test[test_idx]
        })

        # Predict for new days (optional, not used in plot)
        X_new = np.arange(30, 91).reshape(-1, 1)
        X_new_scaled = scaler_X.transform(X_new)
        y_new_scaled = sgdr.predict(X_new_scaled)
        y_new = scaler_y.inverse_transform(y_new_scaled.reshape(-1, 1)).flatten()
        return y, dd, y_new

  
def plotPredictions(df,dates):
    actual, dd, _ = SGDregression(df)
    plt.figure()
    plt.plot(dates, actual, label="Actual", color='blue')
    # Map predicted days to dates for plotting
    min_date = dates.min()
    pred_dates = [min_date + pd.Timedelta(days=float(day)) for day in dd['Day']]
    plt.plot(pred_dates, dd['Predicted Price'], label="Predicted", color='red', linestyle='--')
    plt.title("Predicted and Actual Stock Price, 30 Days")
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks(rotation=70)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return dd
    
if __name__ == '__main__':
    # Predictions for 1 month
    filepath = "data\\TSLA.csv"       
    data, dates= formatData(filepath,30)
    plotPredictions(data,dates)
