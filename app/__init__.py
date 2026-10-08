"""
Tanim Guard Flask Application Factory
"""
import os
from flask import Flask
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv

csrf = CSRFProtect()

def create_app():
    """Create and configure the Flask application"""
    load_dotenv()

    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
    app.config['DATABASE_URL'] = os.getenv('DATABASE_URL', 'sqlite:///tanim_guard.db')
    app.config['DEFAULT_LANGUAGE'] = os.getenv('DEFAULT_LANGUAGE', 'tl')

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize extensions
    csrf.init_app(app)

    # Initialize database
    from app.models import init_db
    init_db(app)

    # Register routes
    from app import routes
    app.register_blueprint(routes.bp)

    return app
