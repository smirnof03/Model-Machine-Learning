from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')

# Nama-nama tanaman (sesuaikan dengan dataset)
crop_names = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
    'apple', 'orange', 'papaya', 'coconut', 'cotton',
    'jute', 'coffee'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        data = request.get_json()
        
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        # Buat array fitur
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Scaling
        features_scaled = scaler.transform(features)
        
        # Prediksi
        prediction = model.predict(features_scaled)
        
        # Ambil probabilitas
        probabilities = model.predict_proba(features_scaled)
        confidence = np.max(probabilities) * 100
        
        result = {
            'success': True,
            'crop': prediction[0],
            'confidence': round(confidence, 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)