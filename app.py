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
    img = cv2.imread(filepath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask_3channel = cv2.merge([mask, mask, mask])

    result = cv2.bitwise_and(img, mask_3channel)
    enhanced_img = cv2.convertScaleAbs(result, alpha=1.5, beta=0)

    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(filepath))
    cv2.imwrite(processed_filepath, enhanced_img)

    return processed_filepath

if __name__ == '__main__':
    app.run(debug=True)
