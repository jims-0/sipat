"""
Tanim Guard Database Models
"""
import sqlite3
from datetime import datetime
from flask import g, current_app

def get_db():
    """Get database connection"""
    if 'db' not in g:
        db_path = current_app.config['DATABASE_URL'].replace('sqlite:///', '')
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Close database connection"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db(app):
    """Initialize database schema"""
    @app.teardown_appcontext
    def teardown_db(exception):
        close_db()

    with app.app_context():
        db = get_db()
        db.executescript('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                disease_name_tl TEXT,
                confidence REAL NOT NULL,
                crop_type TEXT NOT NULL,
                gradcam_path TEXT,
                treatment TEXT,
                treatment_tl TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feedback INTEGER  -- 1=helpful, 0=not helpful, NULL=no feedback
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp ON diagnoses(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_crop_type ON diagnoses(crop_type);
        ''')
        db.commit()

def save_diagnosis(image_path, disease_name, disease_name_tl, confidence,
                   crop_type, gradcam_path, treatment, treatment_tl):
    """Save a diagnosis to the database"""
    db = get_db()
    cursor = db.execute('''
        INSERT INTO diagnoses
        (image_path, disease_name, disease_name_tl, confidence, crop_type,
         gradcam_path, treatment, treatment_tl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (image_path, disease_name, disease_name_tl, confidence, crop_type,
          gradcam_path, treatment, treatment_tl))
    db.commit()
    return cursor.lastrowid

def get_diagnosis_history(limit=50):
    """Get recent diagnosis history"""
    db = get_db()
    rows = db.execute('''
        SELECT * FROM diagnoses
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    return [dict(row) for row in rows]

def get_diagnosis_by_id(diagnosis_id):
    """Get a single diagnosis by ID"""
    db = get_db()
    row = db.execute('SELECT * FROM diagnoses WHERE id = ?', (diagnosis_id,)).fetchone()
    return dict(row) if row else None

def save_feedback(diagnosis_id, helpful):
    """Save user feedback for a diagnosis"""
    db = get_db()
    db.execute('UPDATE diagnoses SET feedback = ? WHERE id = ?',
               (1 if helpful else 0, diagnosis_id))
    db.commit()

def get_stats():
    """Get usage statistics"""
    db = get_db()

    total = db.execute('SELECT COUNT(*) as count FROM diagnoses').fetchone()['count']

    by_crop = db.execute('''
        SELECT crop_type, COUNT(*) as count
        FROM diagnoses
        GROUP BY crop_type
        ORDER BY count DESC
    ''').fetchall()

    by_disease = db.execute('''
        SELECT disease_name, COUNT(*) as count
        FROM diagnoses
        GROUP BY disease_name
        ORDER BY count DESC
        LIMIT 10
    ''').fetchall()

    accuracy = db.execute('''
        SELECT
            AVG(CASE WHEN feedback = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
            COUNT(CASE WHEN feedback IS NOT NULL THEN 1 END) as rated_count
        FROM diagnoses
    ''').fetchone()

    return {
        'total_diagnoses': total,
        'by_crop': [dict(row) for row in by_crop],
        'by_disease': [dict(row) for row in by_disease],
        'user_accuracy': accuracy['accuracy'] if accuracy['rated_count'] > 0 else None,
        'feedback_count': accuracy['rated_count']
    }
