
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def test_model():
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').values.ravel()
    model = joblib.load('linear_regression_model.pkl')
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Test Metrics: MSE: {mse}, R2: {r2}")

if __name__ == "__main__":
    test_model()
