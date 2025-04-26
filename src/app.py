from flask import Flask
from flask_cors import CORS
from src.routes import recommendation_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(recommendation_bp)

if __name__ == '__main__':
    app.run(debug=True)