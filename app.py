import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cache for loaded models
loaded_models = {}

# CIFAR-10 labels
labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Store prediction history in-memory
prediction_history = []

def get_model(model_name):
    """Load and cache model"""
    if model_name not in loaded_models:
        model_path = os.path.join(MODELS_FOLDER, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found")
        loaded_models[model_name] = tf.keras.models.load_model(model_path)
    return loaded_models[model_name]

def preprocess_and_predict(image_path, model_name='gpt-alpha.h5'):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    img_array = image.img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = get_model(model_name)
    predictions = model.predict(img_array)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])

    return {
        'class_id': predicted_class,
        'label': labels[predicted_class],
        'confidence': round(confidence * 100, 2),
        'probabilities': {
            labels[i]: round(float(prob) * 100, 2)
            for i, prob in enumerate(predictions)
        },
        'model_used': model_name
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    try:
        model_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.h5')]
        models = []
        for model_file in model_files:
            model_path = os.path.join(MODELS_FOLDER, model_file)
            size_bytes = os.path.getsize(model_path)
            size_mb = round(size_bytes / (1024 * 1024), 1)
            models.append({
                'filename': model_file,
                'size_mb': size_mb,
                'size_bytes': size_bytes
            })
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    return jsonify(prediction_history)

@app.route('/predictions', methods=['POST'])
def create_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Get selected model (default to gpt-beta.h5)
    model_name = request.form.get('model', 'gpt-beta.h5')

    # Validate model exists
    model_path = os.path.join(MODELS_FOLDER, model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 400

    # Save file to static/uploads with a timestamped name
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        result = preprocess_and_predict(file_path, model_name)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Generate URL for the saved image
    image_url = f"/static/uploads/{filename}"

    record = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'filename': filename,
        'image_url': image_url,
        **result
    }
    prediction_history.append(record)
    return jsonify(record), 201

if __name__ == '__main__':
    host = '0.0.0.0'
    port = int(os.getenv('PORT', 10000))
    app.run(host=host, port=port, debug=False)
