import pandas as pd
import joblib
from src.features import load_timeseries, build_lag_features

def prepare_for_app(data_path, n_lags=3):
    df = load_timeseries(data_path)
    df = build_lag_features(df, n_lags=n_lags)
    df = df.dropna(subset=[f"lag_{i}" for i in range(1, n_lags+1)])
    return df

def predict_next_day(model, df_state, n_lags=3):
    df_state = df_state.sort_values("Date")
    last = df_state.tail(n_lags)
    if len(last) < n_lags:
        return None

    
    lags = [int(last["NewCases"].iloc[-i]) for i in range(1, n_lags+1)]
    ma_3 = float(last["NewCases"].rolling(3, min_periods=1).mean().iloc[-1])
    X = [lags + [ma_3]]

    
    pred = int(model.predict(X)[0])

    
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
        prob = min(prob, 0.99)  
    else:
        prob = None

    return {"pred": pred, "prob": prob}
