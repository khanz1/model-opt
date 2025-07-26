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
print(f"Creating upload directory: {UPLOAD_FOLDER}")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(f"Upload folder configured: {UPLOAD_FOLDER}")

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
        print(f"Loading model: {model_name}")
        model_path = os.path.join(MODELS_FOLDER, model_name)
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            raise FileNotFoundError(f"Model {model_name} not found")
        
        print(f"Loading TensorFlow model from: {model_path}")
        loaded_models[model_name] = tf.keras.models.load_model(model_path)
        print(f"Model cached successfully: {model_name}")
    else:
        print(f"Using cached model: {model_name}")
    
    return loaded_models[model_name]

def preprocess_and_predict(image_path, model_name='gpt-alpha.h5'):
    print(f"Starting image preprocessing: {image_path}")
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    print(f"Original image size: {img.size}")
    
    img = img.resize((32, 32))
    print(f"Resized to CIFAR-10 format: 32x32")
    
    img_array = image.img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    print(f"Image preprocessed - shape: {img_array.shape}")

    # Get model and predict
    model = get_model(model_name)
    print(f"Running prediction with model: {model_name}")
    
    predictions = model.predict(img_array)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])
    
    print(f"Prediction complete - Class: {labels[predicted_class]}, Confidence: {confidence*100:.2f}%")

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
    print("GET / - Serving main page")
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    print(f"GET /static/uploads/{filename} - Serving uploaded file")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    print("GET /models - Listing available models")
    try:
        model_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.h5')]
        print(f"Found {len(model_files)} model files: {model_files}")
        
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
            print(f"  - {model_file}: {size_mb}MB")
        
        print(f"Returning {len(models)} models")
        return jsonify(models)
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    print(f"GET /predictions - Returning {len(prediction_history)} prediction records")
    return jsonify(prediction_history)

@app.route('/predictions', methods=['POST'])
def create_prediction():
    print("POST /predictions - New prediction request received")
    
    if 'image' not in request.files:
        print("No image file in request")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        print("Empty filename provided")
        return jsonify({'error': 'Empty filename'}), 400

    # Get selected model (default to gpt-beta.h5)
    model_name = request.form.get('model', 'gpt-beta.h5')
    print(f"Selected model: {model_name}")

    # Validate model exists
    model_path = os.path.join(MODELS_FOLDER, model_name)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return jsonify({'error': f'Model {model_name} not found'}), 400

    # Save file to static/uploads with a timestamped name
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    print(f"Saving uploaded file: {filename}")
    file.save(file_path)
    print(f"File saved to: {file_path}")

    try:
        print(f"Starting prediction process...")
        result = preprocess_and_predict(file_path, model_name)
        print(f"Prediction successful: {result['label']} ({result['confidence']}%)")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
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
    
    print(f"Added to prediction history (total: {len(prediction_history)} records)")
    print(f"Prediction complete - returning result")
    
    return jsonify(record), 201

if __name__ == '__main__':
    host = '0.0.0.0'
    port = int(os.getenv('PORT', 10000))
    
    print(f"Starting Flask server on {host}:{port}")
    
    app.run(host=host, port=port, debug=False)
