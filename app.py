from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('breakout_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        input_df = pd.DataFrame([data])
        input_df = input_df[features]
        
        input_scaled = scaler.transform(input_df)
        
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'breakout_probability': float(probability),
            'prediction': 'Breakout' if prediction == 1 else 'No Breakout',
            'player_name': data.get('Name', 'Unknown')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)