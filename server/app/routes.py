from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from .pdf_analyser import pdf_to_json
import os 

FILE_DIRECTORY = os.path.join(os.path.dirname(__file__), 'tmp')
server = Blueprint('server', __name__)

@server.route("/health", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify({"status": "ok", "message": "Server is healthy"}), 200

@server.route('/hello', methods=['GET'])
@cross_origin()
def hello():
    return 'Hello, World!'

@server.route('/analyze', methods=['POST'])
@cross_origin()
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        path = os.path.join(FILE_DIRECTORY, file.filename) 
        with open(path, 'wb') as f:
            f.write(file.read())
        result = pdf_to_json(path)
        os.remove(path)
        return jsonify(result), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
            