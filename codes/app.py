from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
import gridfs
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS  # Import CORS
from model import load_model_and_verify
from bson import ObjectId  
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from flask import send_from_directory

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# MongoDB configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/signatureDB'
mongo = PyMongo(app)
fs = gridfs.GridFS(mongo.db)

# Directory for saving uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
print(app.config['UPLOAD_FOLDER'])
def save_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path

UPLOAD_FOLD = 'C:/Users/pshab/Desktop/signature verification/signver/backend/uploads'
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLD, filename)

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    file = request.files.get('file')

    if not username or not email or not password or not file:
        return jsonify({'error': 'Missing fields or file'}), 400

    # Save file to disk and get file path
    file_path = save_file(file)

    # Store the file in GridFS
    with open(file_path, 'rb') as img_file:
        file_id = fs.put(img_file, filename=file.filename, metadata={'username': username})

    # Create a collection with the name of the username
    user_collection = mongo.db[username]

    user_data = {
        'email': email,
        'password': password,  # Note: Passwords should be hashed in production
        'signature_file_id': file_id
    }

    user_collection.insert_one(user_data)

    # Remove the saved file from disk after storing it in GridFS
    os.remove(file_path)

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/verify', methods=['POST'])
def verify():
    username = request.form.get('username')
    uploaded_file = request.files.get('file')

    if not username or not uploaded_file:
        return jsonify({'error': 'Username and file are required'}), 400
    
    user_collection = mongo.db.get_collection(username)  # Collection named after the username
    if user_collection is None:
        return jsonify({'error': 'User collection not found'}), 404
    
    user_document = user_collection.find_one()
    if not user_document:
        return jsonify({'error': 'User not found or invalid type'}), 404
    
    # Retrieve the original signature file from GridFS using signature_file_id
    signature_file_id = user_document.get('signature_file_id')
    if not signature_file_id:
        return jsonify({'error': 'No signature file found for user'}), 404
    
    # Convert signature_file_id to ObjectId
    signature_file_id = ObjectId(signature_file_id['$oid']) if isinstance(signature_file_id, dict) else ObjectId(signature_file_id)
    
    try:
        original_signature_file = fs.get(signature_file_id)
    except gridfs.errors.NoFile:
        return jsonify({'error': 'Original signature file not found'}), 404

    # Save the original signature file temporarily
    original_signature_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_signature.png')
    with open(original_signature_path, 'wb') as f:
        f.write(original_signature_file.read())

    # Save the uploaded file temporarily
    uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
    uploaded_file.save(uploaded_file_path)

    # Send the original signature and the uploaded file to model.py for verification
    verification_result,simind = load_model_and_verify(original_signature_path, uploaded_file_path)
    euclid_value = float(simind)
    
    # Calculate the similarity index
    simind = round(euclid_value,3)
    original_signature_path= f'http://localhost:5000/uploads/{os.path.basename(original_signature_path)}'
    uploaded_file_path = f'http://localhost:5000/uploads/{os.path.basename(uploaded_file_path)}'


    #try:
     #   os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'original_signature.png'))
      #  os.remove(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename)))
    #except OSError as e:
     #   print(f"Error removing file: {e}")
    print(original_signature_path)
    print(uploaded_file_path)
    return jsonify({'prediction': verification_result,'simind':simind,'org':original_signature_path,'uploaded':uploaded_file_path})
if __name__ == '__main__':
    app.run(debug=True)
