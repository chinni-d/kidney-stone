from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import sys
import io

# Set the default encoding for stdout to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model_path = './model/mlp_model.h5'
model = load_model(model_path)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded image to the static folder
        img_path = os.path.join('./static', file.filename)
        file.save(img_path)

        # Process the image for prediction
        img = load_img(img_path, target_size=(64, 64))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match input shape
        
        # Predict using the model
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        class_label = 'Stone Detected' if class_index == 0 else 'No Stone'
        
        # Redirect to result page with prediction result
        return redirect(url_for('result', filename=file.filename, label=class_label))

# Route for result page
@app.route('/result')
def result():
    filename = request.args.get('filename')
    label = request.args.get('label')
    return render_template('result.html', filename=filename, label=label)

if __name__ == '__main__':
    app.run(debug=True)
