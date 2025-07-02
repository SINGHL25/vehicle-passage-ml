import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

def train_model(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    X = df[['hour', 'dayofweek']]
    y = df['vehicle_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    return model, y_test, y_pred

def predict_next(model, hour, dayofweek):
    return model.predict([[hour, dayofweek]])[0]

def plot_prediction(y_test, y_pred, out_file):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:50], label='Actual', marker='o')
    plt.plot(y_pred[:50], label='Predicted', marker='x')
    plt.title('Actual vs Predicted Vehicle Counts')
    plt.xlabel('Sample')
    plt.ylabel('Vehicle Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
