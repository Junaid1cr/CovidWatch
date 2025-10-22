

import pandas as pd

def load_timeseries(path: str) -> pd.DataFrame:
    """
    Load the COVID-19 India statewise timeseries CSV and preprocess it.
    Returns a DataFrame with columns: Date, State, Confirmed, Deaths, Recovered, Active, NewCases
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    
    
    df.columns = [c.strip() for c in df.columns]
    
    
    df = df.rename(columns={
        "State/Province": "State",
        "Province/State": "State",
        "Confirmed": "Confirmed",
        "Deaths": "Deaths",
        "Recovered": "Recovered"
    })
    
    
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    
   
    df[["Confirmed", "Deaths", "Recovered"]] = df[["Confirmed", "Deaths", "Recovered"]].fillna(0).astype(int)
    
    
    df["Active"] = df["Confirmed"] - df["Deaths"] - df["Recovered"]
    
    
    df["NewCases"] = df.groupby("State")["Confirmed"].diff().fillna(0).astype(int)
    
    return df


def build_lag_features(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    """
    Add lag features (lag_1, lag_2, ..., lag_n) and 3-day moving average (ma_3)
    """
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df.groupby("State")["NewCases"].shift(lag)
    
    
    df["ma_3"] = df.groupby("State")["NewCases"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    
    return df


def create_target(df: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    """
    Create a target column indicating whether next day's new cases exceed the threshold.
    target = 1 if next_day_new > threshold else 0
    """
    df = df.copy()
    df["next_day_new"] = df.groupby("State")["NewCases"].shift(-1)
    df["target"] = (df["next_day_new"] > threshold).astype(int)
    return df
