
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def prepare_data():
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data['data'], data['target'], test_size=0.2, random_state=42
    )
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('y_test.csv', index=False)
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
