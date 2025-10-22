
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
from datetime import datetime
from src.features import load_timeseries, build_lag_features
import traceback

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


MODEL_PATH = "models/rf_model.joblib"
DATA_PATH = "data/india_statewise_timeseries.csv"


try:
    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None


data_cache = {}

def load_state_data(state):
    """Load and prepare data for a specific state using your feature pipeline"""
    if state in data_cache:
        return data_cache[state]
    
    try:
        
        df = load_timeseries(DATA_PATH)
        
        
        state_normalized = state.replace('_', ' ').title()
        
        print(f"Available states: {df['State'].unique().tolist()}")
        print(f"Looking for: {state_normalized}")
        
        
        df_state = df[df['State'].str.lower() == state_normalized.lower()]
        
        if df_state.empty:
            print(f"State '{state_normalized}' not found in data")
            return None
        
        print(f"✓ Found state: {state_normalized}, shape: {df_state.shape}")
        print(f"Columns: {df_state.columns.tolist()}")
        
        data_cache[state] = df_state
        return df_state
    except Exception as e:
        print(f"Error loading state data: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def home():
    """Serve the main index.html"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict COVID cases for a specific state and time"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        state = data.get('state', '').lower().strip()
        month = data.get('month', 10)
        year = data.get('year', 2025)
        n_lags = data.get('n_lags', 3)
        
        print(f"Received: state={state}, month={month}, year={year}, n_lags={n_lags}")
        
        if not state:
            return jsonify({"error": "State is required"}), 400
        
        
        df_state = load_state_data(state)
        if df_state is None:
            print(f"State not found: {state}")
            return jsonify({"error": f"State '{state}' not found in database"}), 404
        
        print(f"Loaded data for {state}, shape: {df_state.shape}")
        
        
        df_state = df_state.sort_values('Date') if 'Date' in df_state.columns else df_state
        df_prepared = df_state.copy()
        
        print(f"DataFrame columns: {df_prepared.columns.tolist()}")
        
        if 'NewCases' not in df_prepared.columns:
            print(f"ERROR: NewCases column not found! Creating it now...")
            
            return jsonify({"error": "NewCases column not found in data"}), 400
        
        print(f"NewCases sample: {df_prepared['NewCases'].head()}")
        
        
        df_prepared = build_lag_features(df_prepared, n_lags=n_lags)
        print(f"After build_lag_features: {df_prepared.columns.tolist()}")
        
        
        df_prepared = df_prepared.dropna(subset=[f'lag_{i}' for i in range(1, n_lags + 1)])
        
        print(f"After removing NaN: {len(df_prepared)} rows")
        
        if len(df_prepared) == 0:
            print(f"ERROR: No data after removing NaN")
            return jsonify({"error": "Insufficient historical data"}), 400
        
        
        last = df_prepared.tail(n_lags)
        print(f"Last rows shape: {last.shape}")
        
        
        lags = [int(last["NewCases"].iloc[-i]) for i in range(1, n_lags+1)]
        print(f"Lags extracted: {lags}")
        
        
        ma_3 = float(last["NewCases"].rolling(3, min_periods=1).mean().iloc[-1])
        print(f"Moving average: {ma_3}")
        
        
        X = [lags + [ma_3]]
        print(f"Features for prediction: {X}")
        
        
        pred = int(model.predict(X)[0])
        pred = max(0, pred)  
        print(f"Raw prediction: {pred}")
        
        
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X)[0, 1])
                prob = min(prob, 0.99)
                print(f"Probability: {prob}")
            except Exception as pe:
                print(f"Could not get probability: {pe}")
                prob = None
        
        result = {
            "pred": pred,
            "prob": prob,
            "state": state,
            "month": month,
            "year": year
        }
        
        print(f"✓ Prediction result: {result}")
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Request error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/bot', methods=['POST'])
def bot_response():
    """Bot Q&A endpoint"""
    try:
        data = request.json
        state = data.get('state', '').lower()
        question = data.get('question', '').lower()
        
        if not state or not question:
            return jsonify({"error": "State and question are required"}), 400
        
        
        answer = get_bot_answer(state, question)
        
        return jsonify({"answer": answer}), 200
    
    except Exception as e:
        print(f"Bot error: {e}")
        return jsonify({"error": str(e)}), 400

def get_bot_answer(state, question):
    """Generate bot answers based on keywords"""
    
    state_name = state.replace('_', ' ').title()
    
    
    if any(word in question for word in ['prevent', 'prevention', 'protect', 'avoid']):
        return f"""
        <strong>COVID-19 Prevention Tips for {state_name}:</strong><br>
        • Wear N95 masks in crowded places<br>
        • Maintain 6 feet distance from others<br>
        • Wash hands frequently with soap for 20 seconds<br>
        • Avoid touching face, eyes, and mouth<br>
        • Stay updated with vaccination status<br>
        • Improve indoor ventilation<br>
        • Stay home if you feel sick<br>
        • Practice respiratory hygiene when coughing/sneezing
        """
    
    
    if any(word in question for word in ['symptom', 'sign', 'fever', 'cough', 'sick']):
        return f"""
        <strong>Common COVID-19 Symptoms:</strong><br>
        • Fever (high temperature)<br>
        • Cough (usually dry)<br>
        • Fatigue or tiredness<br>
        • Loss of taste or smell<br>
        • Difficulty breathing<br>
        • Sore throat<br>
        • Headache<br>
        • Muscle or body aches<br>
        <br>
        Symptoms may appear 2-14 days after exposure. Seek medical help if symptoms worsen.
        """
    
    
    if any(word in question for word in ['hospital', 'treatment', 'doctor', 'medical', 'help']):
        return f"""
        <strong>Treatment & Healthcare in {state_name}:</strong><br>
        • Contact your local health department for guidance<br>
        • Seek immediate medical care if experiencing severe symptoms<br>
        • Call emergency services (911/112) for critical cases<br>
        • Home isolation is recommended for mild cases<br>
        • Stay hydrated and rest<br>
        • Consult healthcare provider before taking medication<br>
        • Follow local quarantine guidelines
        """
    
    
    if any(word in question for word in ['cause', 'spread', 'transmit', 'how', 'virus']):
        return f"""
        <strong>COVID-19 Causes & Transmission:</strong><br>
        • Caused by SARS-CoV-2 virus<br>
        • Spreads through respiratory droplets from coughs/sneezes<br>
        • Can spread when breathing in air near infected person<br>
        • Surfaces transmission is less common<br>
        • Airborne transmission in poorly ventilated spaces<br>
        • Contact with infected persons increases risk<br>
        • Incubation period: typically 2-14 days
        """
    
    
    if any(word in question for word in ['vaccine', 'vaccination', 'immunize', 'boost']):
        return f"""
        <strong>COVID-19 Vaccination in {state_name}:</strong><br>
        • Vaccination is the most effective prevention<br>
        • Multiple vaccines available (Covaxin, Covishield, etc.)<br>
        • Free vaccination available at government centers<br>
        • Follow recommended vaccination schedules<br>
        • Booster doses available for eligible groups<br>
        • Side effects are generally mild<br>
        • Consult health officials for eligibility and nearest vaccination center
        """
    
    
    return f"""
    I can help with information about COVID-19 in {state_name}. 
    Try asking about:<br>
    • Prevention measures<br>
    • Symptoms to watch for<br>
    • Treatment options<br>
    • Transmission methods<br>
    • Vaccination details<br>
    <br>
    Ask me any specific question!
    """

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/api/states', methods=['GET'])
def get_states():
    """Get list of available states"""
    try:
        df = pd.read_csv(DATA_PATH)
        states = df['State'].unique().tolist() if 'State' in df.columns else []
        return jsonify({"states": states}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)