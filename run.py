"""
Tanim Guard - Development server entry point
Run with: python run.py
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',  # Accessible from other devices on network (for mobile testing)
        port=5000,
        debug=True
    )
