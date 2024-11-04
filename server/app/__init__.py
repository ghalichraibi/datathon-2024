from flask import Flask
from .routes import server

def create_app():
    app = Flask(__name__)
    
    # Register the server blueprint
    app.register_blueprint(server)
    
    return app