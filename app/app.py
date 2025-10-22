import argparse
import joblib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from ..utils import prepare_for_app, predict_next_day

def create_app(data_path, model_path, n_lags=3):
    df = prepare_for_app(data_path, n_lags=n_lags)
    model = joblib.load(model_path)

    app = dash.Dash(__name__)
    states = sorted(df["State"].unique())

    app.layout = html.Div([
        html.H1("India COVID-19 Dashboard"),
        dcc.Dropdown(id="state-dd", options=[{"label": s, "value": s} for s in states], value=states[0]),
        dcc.Graph(id="time-series"),
        html.Div(id="prediction")
    ], style={"maxWidth":"900px","margin":"auto"})

    @app.callback([Output("time-series","figure"), Output("prediction","children")],
                  [Input("state-dd","value")])
    def update(state):
        dsub = df[df["State"]==state].sort_values("Date")
        fig = px.line(dsub, x="Date", y=["Confirmed","NewCases"], title=f"{state} — Confirmed & New Cases")
        pr = predict_next_day(model, dsub, n_lags=n_lags)
        if pr is None:
            pred_text = html.P("Not enough history to predict.")
        else:
            text = "Positive (increase)" if pr["pred"]==1 else "Negative (no significant increase)"
            pred_text = html.Div([html.H3("Next-day prediction: "+text),
                                  html.P(f"Probability: {pr['prob']:.2f}" if pr["prob"] else "")])
        return fig, pred_text

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    application = create_app(args.data, args.model)
    application.run(debug=True, port=args.port)

