from flask import Flask
from flask_cors import CORS

from .routes import server


def create_app():
    app = Flask(__name__)
    
    # Enable CORS for all routes, allowing requests from any domain.
    CORS(app)    

    app.register_blueprint(server)
    
    return app