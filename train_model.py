
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

def train_model():
    mlflow.set_experiment("California Housing Experiment")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').values.ravel()

    model = LinearRegression()

    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)
        joblib.dump(model, 'linear_regression_model.pkl')

        # Make predictions and calculate metrics
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log parameters, metrics, and artifacts
        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_artifact('linear_regression_model.pkl')

        print(f"Training complete. MSE: {mse}, R2: {r2}")

if __name__ == "__main__":
    train_model()
