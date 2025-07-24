import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once
model = tf.keras.models.load_model('lightweight_model.h5')

# CIFAR-10 labels
labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Store prediction history in-memory
prediction_history = []

def preprocess_and_predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    img_array = image.img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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
        }
    }

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

    # Save file to static/uploads with a timestamped name
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    result = preprocess_and_predict(file_path)

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
