"""
Tanim Guard Flask Application Factory
"""
import os
from flask import Flask
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv

csrf = CSRFProtect()

def create_app(overrides=None):
    """Create and configure the Flask application.

    `overrides` (tests) is applied last, e.g. a temporary DATABASE_URL and
    UPLOAD_FOLDER so a test run never touches tanim_guard.db.
    """
    load_dotenv()

    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = _secret_key(app)
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
    app.config['DATABASE_URL'] = os.getenv('DATABASE_URL', 'sqlite:///tanim_guard.db')
    app.config['DEFAULT_LANGUAGE'] = os.getenv('DEFAULT_LANGUAGE', 'tl')
    app.config.update(overrides or {})

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

    app.add_template_filter(markdown_lite, 'markdown')
    return app


def _secret_key(app):
    """SECRET_KEY from .env if it is a real one; otherwise one random key per
    install, kept in instance/secret_key. The shipped placeholder must never be
    the key - anyone who has read .env.example could forge sessions with it."""
    import secrets

    configured = os.getenv('SECRET_KEY', '')
    if configured and 'change' not in configured and 'your-secret' not in configured:
        return configured
    path = os.path.join(app.instance_path, 'secret_key')
    if not os.path.exists(path):
        os.makedirs(app.instance_path, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(secrets.token_hex(32))
    with open(path, encoding='utf-8') as handle:
        return handle.read().strip()


def markdown_lite(text):
    """The small Markdown subset treatment.py writes (**bold**, _italic_,
    numbered lines, paragraphs) -> HTML. Escapes first, so nothing in the text
    can inject markup."""
    import re
    from markupsafe import Markup, escape

    html, in_list = [], False
    for line in str(escape(text or '')).split('\n'):
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        line = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<em>\1</em>', line)
        item = re.match(r'\s*\d+\.\s+(.*)', line)
        if item:
            if not in_list:
                html.append('<ol>')
                in_list = True
            html.append(f'<li>{item.group(1)}</li>')
            continue
        if in_list:
            html.append('</ol>')
            in_list = False
        if line.strip():
            html.append(f'<p>{line}</p>')
    if in_list:
        html.append('</ol>')
    return Markup('\n'.join(html))
