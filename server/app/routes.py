from flask import Blueprint, jsonify, request

server = Blueprint('server', __name__)

@server.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is healthy"}), 200

@server.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'

@server.route('/analyze', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf']
    
    print(request.files)
    print(file)
    print(file.filename)

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    return jsonify({'message': 'File uploaded successfully'}), 200            
            