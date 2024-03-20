from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('Densenet_model.keras')

# Define class names
class_names = ['Tampered', 'Authentic']

# Function to prepare the image for prediction
def prepare_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = image.reshape(-1, 128, 128, 3)
    return image

# Flask route to render the upload form
@app.route('/')
def upload_form():
    return render_template('index.html')

# Flask route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file uploaded')
        
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', prediction='No file selected')

        # Check if the file is valid
        if file:
            try:
                # Save the file to a temporary location
                file_path = 'static/temp.jpg'
                file.save(file_path)

                # Prepare the image for prediction
                image = prepare_image(file_path)

                # Make prediction
                prediction = model.predict(image)
                prediction_class = np.argmax(prediction, axis=1)[0]
                confidence = np.amax(prediction) * 100
                result = {
                    'class': class_names[prediction_class],
                    'confidence': confidence,
                    'image_path': file_path
                }

                return render_template('result.html', result=result)

            except Exception as e:
                return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
