from flask import Flask, render_template, request, jsonify
import os
import traceback
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.utils import custom_object_scope
from werkzeug.utils import secure_filename

app = Flask(__name__)

def custom_f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))

MODEL_PATH = 'LicensePlateModel.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
with custom_object_scope({'custom_f1score': custom_f1score}):
    model = load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            print(f"File uploaded: {filepath}")  
            # Preprocess the image to match model's expected input shape
            img = image.load_img(filepath, target_size=(28, 28))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            print(f"Input shape to model: {img_array.shape}")

            predictions = model.predict(img_array)
            print(f"Predictions: {predictions}")

            class_labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

            top_n = 9
            predicted_labels = []

            for i in range(predictions.shape[1]):  
                top_indices = np.argsort(predictions[0, i, :])[-top_n:][::-1]
                top_probs = predictions[0, i, top_indices]
                top_classes = [class_labels[idx] for idx in top_indices]
                predicted_labels.append((top_classes, top_probs))

            for i, (classes, probs) in enumerate(predicted_labels):
                print(f"Character {i+1}:")
                for cls, prob in zip(classes, probs):
                    print(f"  {cls}: {prob:.4f}")

            os.remove(filepath)

            return jsonify({'predicted_labels': predicted_labels})
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred', 'message': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  
    app.run(debug=True)
