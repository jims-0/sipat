"""
Sipat Routes
"""
import json
import os
import uuid
from datetime import datetime

from flask import (Blueprint, current_app, jsonify, render_template, request,
                   send_from_directory)
from PIL import Image, ImageOps, UnidentifiedImageError

from app.ml.adviser import ask
from app.ml.gradcam import generate_gradcam
from app.ml.inference import predict_disease
from app.ml.treatment import get_treatment_guidance
from app.models import (get_diagnosis_by_id, get_diagnosis_history, get_stats,
                        save_diagnosis, save_feedback)

bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/')
def index():
    """Main page - camera/upload interface"""
    return render_template('index.html')


@bp.route('/api/predict', methods=['POST'])
def predict():
    """Photo in, diagnosis out (saved to history)."""
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify({'error': 'Walang larawan na na-upload'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Mali ang format ng larawan. Gumamit ng JPG o PNG.'}), 400

    # Our own name, never the uploaded one: no collisions, nothing user-controlled on disk.
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        with Image.open(file.stream) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            img.save(filepath, 'JPEG', quality=85, optimize=True)
    except (UnidentifiedImageError, OSError):
        return jsonify({'error': 'Hindi mabasa ang larawan. Subukang kumuha ulit.'}), 400

    prediction = predict_disease(filepath)
    if prediction['status'] == 'error':
        os.remove(filepath)
        return jsonify({'error': prediction['error']}), 503

    gradcam_path = None
    if prediction['status'] in ('ok', 'unsure'):
        gradcam_filename = f"gradcam_{filename}"
        if generate_gradcam(filepath, os.path.join(current_app.config['UPLOAD_FOLDER'],
                                                   gradcam_filename), prediction['class_idx']):
            gradcam_path = f"/uploads/{gradcam_filename}"

    treatment = get_treatment_guidance(prediction['class_key'], prediction['status'])
    diagnosis_id = save_diagnosis(
        image_path=f"/uploads/{filename}",
        disease_name=prediction['disease_name'],
        disease_name_tl=prediction['disease_name_tl'],
        confidence=prediction['confidence'],
        crop_type=prediction['crop_type'],
        gradcam_path=gradcam_path,
        treatment=treatment['en'],
        treatment_tl=treatment['tl'],
        status=prediction['status'],
        class_key=prediction['class_key'],
        top3_json=json.dumps(prediction['top3']),
    )
    language = request.form.get('language', current_app.config['DEFAULT_LANGUAGE'])
    return jsonify({
        'success': True,
        'diagnosis_id': diagnosis_id,
        'status': prediction['status'],
        'disease_name': prediction['disease_name'],
        'disease_name_tl': prediction['disease_name_tl'],
        'confidence': prediction['confidence'],
        'crop_type': prediction['crop_type'],
        'top3': prediction['top3'],
        'image_url': f"/uploads/{filename}",
        'gradcam_url': gradcam_path,
        'treatment': treatment.get(language, treatment['en']),
    })


@bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Photos live in UPLOAD_FOLDER, which in the desktop build is a folder next
    to the exe - never inside the bundle, which is deleted when the app closes."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@bp.route('/history')
def history():
    """View diagnosis history"""
    diagnoses = get_diagnosis_history(limit=50)
    return render_template('history.html', diagnoses=diagnoses)


@bp.route('/diagnosis/<int:diagnosis_id>')
def view_diagnosis(diagnosis_id):
    """View a single diagnosis in detail"""
    diagnosis = get_diagnosis_by_id(diagnosis_id)
    if not diagnosis:
        return render_template('about.html'), 404
    diagnosis['top3'] = json.loads(diagnosis.get('top3_json') or '[]')
    return render_template('result.html', diagnosis=diagnosis)


@bp.route('/api/ask/<int:diagnosis_id>', methods=['POST'])
def ask_adviser(diagnosis_id):
    """Follow-up question about one diagnosis, answered by the local adviser."""
    diagnosis = get_diagnosis_by_id(diagnosis_id)
    if not diagnosis:
        return jsonify({'error': 'Hindi makita ang pagsusuri'}), 404
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({'error': 'Walang tanong'}), 400
    return jsonify(ask(diagnosis, question, data.get('lang', 'tl')))


@bp.route('/api/feedback/<int:diagnosis_id>', methods=['POST'])
def submit_feedback(diagnosis_id):
    """Submit user feedback on diagnosis accuracy"""
    data = request.get_json(silent=True) or {}
    helpful = data.get('helpful', None)

    if helpful is None:
        return jsonify({'error': 'Missing feedback value'}), 400

    save_feedback(diagnosis_id, helpful)
    return jsonify({'success': True})


@bp.route('/stats')
def stats():
    """View usage statistics"""
    statistics = get_stats()
    return render_template('stats.html', stats=statistics)


@bp.route('/about')
def about():
    """About page - with the measured model numbers, read from the report."""
    from app.ml.inference import MODEL_DIR
    report_path = MODEL_DIR / 'evaluation_report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return render_template('about.html', report=report)
