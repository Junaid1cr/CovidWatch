import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from .features import load_timeseries, build_lag_features, create_target

def main(data_path, model_out, threshold=50, n_lags=3):
    df = load_timeseries(data_path)
    df = build_lag_features(df, n_lags=n_lags)
    df = create_target(df, threshold=threshold)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags+1)] + ["ma_3"]
    df = df.dropna(subset=feature_cols + ["target"])

    cutoff_date = df["Date"].quantile(0.8)
    train = df[df["Date"] <= cutoff_date]
    test = df[df["Date"] > cutoff_date]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    rf = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)
    probs = best.predict_proba(X_test)[:, 1]

    print("Best params:", grid.best_params_)
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    print("Logistic Regression report:")
    print(classification_report(y_test, lr.predict(X_test)))

    joblib.dump(best, model_out)
    print("Model saved to", model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model_out", default="models/rf_model.joblib")
    parser.add_argument("--threshold", default=50, type=int)
    parser.add_argument("--n_lags", default=3, type=int)
    args = parser.parse_args()
    main(args.data, args.model_out, args.threshold, args.n_lags)
