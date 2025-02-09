import joblib
from xgboost import XGBRegressor


def load_model():
    # Load the saved model
    xgb_reg_loaded = joblib.load('xgb_reg_model.pkl')
    return xgb_reg_loaded




