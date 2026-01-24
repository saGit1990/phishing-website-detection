import os
from src.config_loader import load_config
from src.data_loader import load_dataset
from src.preprocessor import preprocess_data
from src.train_xgboost import train_xgboost
from src.train_ann import train_ann
from src.train_logistic import train_logistic
 
def run_pipeline():
    config_loader = load_config()

    df = load_dataset(config_loader['data']['file_path'])

    x_train, x_test, y_train, y_test = preprocess_data(
        df=df,
        target_col = config_loader['data']['target_column'],
        test_size = config_loader['data']['test_size'],
        random_state = config_loader['data']['random_state'],
        scaler_path = os.path.join(config_loader['artifacts']['directory'],config_loader['artifacts']['scaler_filename'])
    )

    train_logistic(
        x_train, x_test, y_train, y_test,
        save_path=f"{config_loader['artifacts']['directory']}/{config_loader['artifacts']['log_model_filename']}"
    )

    train_xgboost(
        x_train, x_test, y_train, y_test,
        params=config_loader['xgboost'],
        save_path=f"{config_loader['artifacts']['directory']}/{config_loader['artifacts']['xgb_model_filename']}"
    )

    train_ann(
        x_train, x_test, y_train, y_test,
        params=config_loader['mlp'],
        save_path=f"{config_loader['artifacts']['directory']}/{config_loader['artifacts']['ann_model_filename']}"
    )

    print('\n Pipeline Completed Sucessfully!')