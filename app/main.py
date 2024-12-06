""" Hand Digit Recoginition App"""
from flask import Flask, request, jsonify
from torch_utils import transform_image, get_prediction
import torch


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/predict', methods=['POST'])
def predict ():
   if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error" : "no file"})
        if not allowed_file(file.filename):
            return jsonify({"error" : "format not supported"})
        
        try:
            img_bytes = file.read()
            tesnor = transform_image(img_bytes)
            prediction = get_prediction(tesnor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return jsonify(data)
        
        except:
            return jsonify({'error' : 'error during prediction'})



