"""
Sipat Database Models
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
        # Added with the real model (2026-10-14). ALTER, not a rebuild, so an
        # existing tanim_guard.db keeps its history.
        have = {row['name'] for row in db.execute('PRAGMA table_info(diagnoses)')}
        # 2026-10-19: true_class_key is what the farmer says it really was (the
        # field test set); plot/recheck/parent are the follow-up workflow.
        for column, kind in (('status', "TEXT DEFAULT 'ok'"), ('class_key', 'TEXT'),
                             ('top3_json', 'TEXT'), ('true_class_key', 'TEXT'),
                             ('plot_id', 'INTEGER'), ('notes', 'TEXT'),
                             ('recheck_due', 'DATE'), ('parent_id', 'INTEGER')):
            if column not in have:
                db.execute(f'ALTER TABLE diagnoses ADD COLUMN {column} {kind}')
        db.executescript('''
            CREATE TABLE IF NOT EXISTS plots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_plot ON diagnoses(plot_id);
        ''')
        db.commit()

def save_diagnosis(image_path, disease_name, disease_name_tl, confidence,
                   crop_type, gradcam_path, treatment, treatment_tl,
                   status='ok', class_key=None, top3_json=None, plot_id=None,
                   parent_id=None):
    """Save a diagnosis to the database"""
    db = get_db()
    cursor = db.execute('''
        INSERT INTO diagnoses
        (image_path, disease_name, disease_name_tl, confidence, crop_type,
         gradcam_path, treatment, treatment_tl, status, class_key, top3_json,
         plot_id, parent_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (image_path, disease_name, disease_name_tl, confidence, crop_type,
          gradcam_path, treatment, treatment_tl, status, class_key, top3_json,
          plot_id, parent_id))
    db.commit()
    return cursor.lastrowid

def get_diagnosis_history(limit=50, plot_id=None, crop=None, status=None, query=None,
                          class_key=None):
    """Recent diagnoses, optionally narrowed to one plot / crop / status, or
    a search on the disease name (English or Tagalog) and the notes."""
    where, args = [], []
    if plot_id:
        where.append('d.plot_id = ?')
        args.append(plot_id)
    if crop:
        where.append('d.crop_type = ?')
        args.append(crop)
    if status:
        where.append("COALESCE(d.status, 'ok') = ?")
        args.append(status)
    if class_key:
        where.append('d.class_key = ?')
        args.append(class_key)
    if query:
        like = f"%{query}%"
        where.append('(d.disease_name LIKE ? OR d.disease_name_tl LIKE ? OR d.notes LIKE ?)')
        args += [like, like, like]
    sql = ('SELECT d.*, p.name AS plot_name FROM diagnoses d LEFT JOIN plots p ON p.id = d.plot_id'
           + (' WHERE ' + ' AND '.join(where) if where else '')
           + ' ORDER BY d.timestamp DESC, d.id DESC LIMIT ?')
    rows = get_db().execute(sql, (*args, limit)).fetchall()
    return [dict(row) for row in rows]

def get_diagnosis_by_id(diagnosis_id):
    """Get a single diagnosis by ID (with its plot name)"""
    db = get_db()
    row = db.execute('SELECT d.*, p.name AS plot_name FROM diagnoses d '
                     'LEFT JOIN plots p ON p.id = d.plot_id WHERE d.id = ?',
                     (diagnosis_id,)).fetchone()
    return dict(row) if row else None

def get_followups(diagnosis_id):
    """Re-scans made from this diagnosis, oldest first."""
    rows = get_db().execute('SELECT * FROM diagnoses WHERE parent_id = ? ORDER BY timestamp, id',
                            (diagnosis_id,)).fetchall()
    return [dict(row) for row in rows]

def delete_diagnosis(diagnosis_id):
    """Remove one diagnosis; returns its row (so the caller can delete the
    photos) or None. Follow-ups made from it stay, unlinked."""
    db = get_db()
    row = get_diagnosis_by_id(diagnosis_id)
    if row:
        db.execute('UPDATE diagnoses SET parent_id = NULL WHERE parent_id = ?', (diagnosis_id,))
        db.execute('DELETE FROM diagnoses WHERE id = ?', (diagnosis_id,))
        db.commit()
    return row

def save_feedback(diagnosis_id, helpful, true_class_key=None):
    """Thumbs up/down, and - when the farmer knows - what it really was.
    A thumbs up on a confident result confirms the model's own answer."""
    db = get_db()
    if helpful and true_class_key is None:
        row = db.execute('SELECT class_key, status FROM diagnoses WHERE id = ?',
                         (diagnosis_id,)).fetchone()
        if row and (row['status'] or 'ok') == 'ok':
            true_class_key = row['class_key']
    db.execute('UPDATE diagnoses SET feedback = ?, true_class_key = ? WHERE id = ?',
               (1 if helpful else 0, true_class_key, diagnosis_id))
    db.commit()

def save_notes(diagnosis_id, notes):
    db = get_db()
    db.execute('UPDATE diagnoses SET notes = ? WHERE id = ?', (notes or None, diagnosis_id))
    db.commit()

def set_recheck(diagnosis_id, due):
    """due: an ISO date string, or None to clear the reminder."""
    db = get_db()
    db.execute('UPDATE diagnoses SET recheck_due = ? WHERE id = ?', (due, diagnosis_id))
    db.commit()

def get_due_rechecks(today):
    """Reminders that are due (or overdue) and not yet answered by a re-scan."""
    rows = get_db().execute('''
        SELECT d.*, p.name AS plot_name FROM diagnoses d
        LEFT JOIN plots p ON p.id = d.plot_id
        WHERE d.recheck_due IS NOT NULL AND d.recheck_due <= ?
          AND NOT EXISTS (SELECT 1 FROM diagnoses c WHERE c.parent_id = d.id)
        ORDER BY d.recheck_due
    ''', (today,)).fetchall()
    return [dict(row) for row in rows]

# --- plots -------------------------------------------------------------------

def get_or_create_plot(name):
    name = (name or '').strip()[:60]
    if not name:
        return None
    db = get_db()
    row = db.execute('SELECT id FROM plots WHERE name = ?', (name,)).fetchone()
    if row:
        return row['id']
    cursor = db.execute('INSERT INTO plots (name) VALUES (?)', (name,))
    db.commit()
    return cursor.lastrowid

def plot_exists(plot_id):
    return get_db().execute('SELECT 1 FROM plots WHERE id = ?', (plot_id,)).fetchone() is not None

def get_plots():
    """Every plot with its scan count and latest result."""
    rows = get_db().execute('''
        SELECT p.id, p.name, COUNT(d.id) AS scans, MAX(d.timestamp) AS last_scan,
               (SELECT COALESCE(l.disease_name_tl, l.disease_name) FROM diagnoses l
                 WHERE l.plot_id = p.id ORDER BY l.timestamp DESC, l.id DESC LIMIT 1) AS last_result
        FROM plots p LEFT JOIN diagnoses d ON d.plot_id = p.id
        GROUP BY p.id ORDER BY p.name
    ''').fetchall()
    return [dict(row) for row in rows]

def delete_plot(plot_id):
    """The plot goes; its scans stay, untagged."""
    db = get_db()
    db.execute('UPDATE diagnoses SET plot_id = NULL WHERE plot_id = ?', (plot_id,))
    db.execute('DELETE FROM plots WHERE id = ?', (plot_id,))
    db.commit()

def get_field_labelled():
    """Scans whose real answer is known (farmer/agronomist label): the
    field test set."""
    rows = get_db().execute('SELECT * FROM diagnoses WHERE true_class_key IS NOT NULL '
                            'ORDER BY id').fetchall()
    return [dict(row) for row in rows]

def get_stats(today=None):
    """Usage statistics. "Field accuracy" compares the model's answer with
    the farmer's own label - only confident ("ok") scans that were labelled."""
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
        WHERE COALESCE(status, 'ok') = 'ok'
        GROUP BY disease_name
        ORDER BY count DESC
        LIMIT 10
    ''').fetchall()

    by_status = db.execute('''
        SELECT COALESCE(status, 'ok') AS status, COUNT(*) AS count
        FROM diagnoses GROUP BY COALESCE(status, 'ok') ORDER BY count DESC
    ''').fetchall()

    # Monday-start weeks, last 8, oldest first; weeks with no scans show as 0.
    from datetime import date, timedelta
    today = today or date.today()
    monday = today - timedelta(days=today.weekday())
    weeks = [monday - timedelta(weeks=i) for i in range(7, -1, -1)]
    counts = {row['week']: row['count'] for row in db.execute('''
        SELECT date(timestamp, '-6 days', 'weekday 1') AS week, COUNT(*) AS count
        FROM diagnoses WHERE date(timestamp) >= ? GROUP BY week
    ''', (weeks[0].isoformat(),))}
    by_week = [{'week': w.isoformat(), 'count': counts.get(w.isoformat(), 0)} for w in weeks]

    accuracy = db.execute('''
        SELECT
            AVG(CASE WHEN feedback = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
            COUNT(*) as rated_count
        FROM diagnoses
        WHERE feedback IS NOT NULL
    ''').fetchone()

    field = db.execute('''
        SELECT COUNT(*) AS labelled,
               SUM(CASE WHEN class_key = true_class_key THEN 1 ELSE 0 END) AS correct
        FROM diagnoses
        WHERE true_class_key IS NOT NULL AND COALESCE(status, 'ok') = 'ok'
    ''').fetchone()
    field_labelled_all = db.execute(
        'SELECT COUNT(*) FROM diagnoses WHERE true_class_key IS NOT NULL').fetchone()[0]

    return {
        'total_diagnoses': total,
        'by_crop': [dict(row) for row in by_crop],
        'by_disease': [dict(row) for row in by_disease],
        'by_status': [dict(row) for row in by_status],
        'by_week': by_week,
        'user_accuracy': accuracy['accuracy'] if accuracy['rated_count'] > 0 else None,
        'feedback_count': accuracy['rated_count'],
        'field_labelled': field['labelled'],
        'field_correct': field['correct'] or 0,
        'field_accuracy': (field['correct'] or 0) / field['labelled'] if field['labelled'] else None,
        'field_labelled_all': field_labelled_all,
    }


def get_dashboard(today=None):
    """Home-screen summary - the desktop twin of dashboard() in pwa/sipat-core.js:
    healthy / sick / unsure counts, rechecks due, scans per week, the most common
    diseases and the latest result of every plot."""
    from datetime import date
    from app.ml.classes import CLASSES

    def kind(status, class_key):
        if (status or 'ok') != 'ok':
            return 'unsure'
        return 'healthy' if CLASSES.get(class_key, {}).get('healthy') else 'sick'

    today = today or date.today()
    db = get_db()
    counts = {'healthy': 0, 'sick': 0, 'unsure': 0}
    diseases = {}
    for row in db.execute('SELECT class_key, status FROM diagnoses'):
        k = kind(row['status'], row['class_key'])
        counts[k] += 1
        if k == 'sick':
            diseases[row['class_key']] = diseases.get(row['class_key'], 0) + 1
    top = [{'key': key, 'count': n, 'name_tl': CLASSES[key]['name_tl'], 'name': CLASSES[key]['name']}
           for key, n in sorted(diseases.items(), key=lambda kv: (-kv[1], kv[0]))[:5] if key in CLASSES]

    plots = []
    for row in db.execute('''
        SELECT p.id AS plot_id, p.name, d.id, d.class_key, d.status, d.timestamp,
               COALESCE(d.disease_name_tl, d.disease_name) AS result
        FROM plots p JOIN diagnoses d ON d.id = (
            SELECT l.id FROM diagnoses l WHERE l.plot_id = p.id ORDER BY l.timestamp DESC, l.id DESC LIMIT 1)
    '''):
        plots.append({**dict(row), 'kind': kind(row['status'], row['class_key'])})
    order = {'sick': 0, 'unsure': 1, 'healthy': 2}
    plots.sort(key=lambda p: (order[p['kind']], p['name'].lower()))

    total = sum(counts.values())
    return {'total': total, **counts, 'due': len(get_due_rechecks(today.isoformat())),
            'top': top, 'weeks': get_stats(today)['by_week'], 'plots': plots}
