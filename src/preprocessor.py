import os 
import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import print_header, ensure_artifacts_dir

def preprocess_data(df, target_col, test_size, random_state, scaler_path):
    print_header("PREPROCESSING DATA")

    df[target_col] = df[target_col].map({-1:0, 1:1})

    x = df.drop([target_col],axis=1)
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x,y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    # create right artifacts
    ensure_artifacts_dir(os.path.dirname(scaler_path))

    # save scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved -> {scaler_path}")

    return x_train_scaled, x_test_scaled, y_train, y_test