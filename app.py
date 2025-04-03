from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        processed_filepath = process_image(filepath)

        return render_template('result.html', original_image=file.filename, processed_image=os.path.basename(processed_filepath))

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

def process_image(filepath):
    # Read the image
    img = cv2.imread(filepath)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Normalize L-channel (IABR enhancement)
    l_min, l_max = np.min(l), np.max(l)
    l = ((l - l_min) / (l_max - l_min) * 255).astype(np.uint8)
    
    # Apply Gaussian blur for smoothing
    l_blurred = cv2.GaussianBlur(l, (5, 5), 0)
    
    # Enhance contrast using weighted addition
    enhanced_l = cv2.addWeighted(l, 1.5, l_blurred, -0.5, 0)
    
    # Merge enhanced L-channel back with A and B channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Apply unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(enhanced_img, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced_img, 1.5, gaussian, -0.5, 0)
    
    # Apply gamma correction
    gamma = 1.2  # Adjust gamma as needed
    gamma_correction = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    final_img = cv2.LUT(sharpened, gamma_correction)
    
    # Save the processed image
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(filepath))
    cv2.imwrite(processed_filepath, final_img)
    
    return processed_filepath

if __name__ == '__main__':
    app.run(debug=True)
