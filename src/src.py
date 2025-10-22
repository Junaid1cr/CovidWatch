import pandas as pd


def load_timeseries(path: str) -> pd.DataFrame:
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
    df["NewCases"] = df.groupby("State")["Confirmed"].diff().fillna(0)
    return df


def build_lag_features(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df.groupby("State")["NewCases"].shift(lag)
    df["ma_3"] = df.groupby("State")["NewCases"].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    return df


def create_target(df: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["next_day_new"] = df.groupby("State")["NewCases"].shift(-1)
    df["target"] = (df["next_day_new"] > threshold).astype(int)
    return df
