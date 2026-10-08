"""
Tanim Guard Routes
"""
import os
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from app.models import (
    save_diagnosis, get_diagnosis_history,
    get_diagnosis_by_id, save_feedback, get_stats
)
from app.ml.inference import predict_disease
from app.ml.gradcam import generate_gradcam
from app.ml.treatment import get_treatment_guidance

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
    """Handle image upload and return diagnosis"""

    # Check if file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'Walang larawan na na-upload'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Walang napiling file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Mali ang format ng larawan. Gumamit ng JPG o PNG.'}), 400

    try:
        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Resize for consistent processing
        img = Image.open(filepath)
        img = img.convert('RGB')
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        img.save(filepath, quality=85, optimize=True)

        # Run ML inference
        prediction = predict_disease(filepath)

        if prediction['error']:
            return jsonify({'error': prediction['error']}), 500

        # Generate Grad-CAM heatmap
        gradcam_filename = f"gradcam_{filename}"
        gradcam_path = os.path.join(current_app.config['UPLOAD_FOLDER'], gradcam_filename)
        generate_gradcam(filepath, gradcam_path, prediction['class_idx'])

        # Get treatment guidance
        language = request.form.get('language', current_app.config['DEFAULT_LANGUAGE'])
        treatment = get_treatment_guidance(
            prediction['disease_name'],
            prediction['crop_type'],
            language
        )

        # Save to database
        diagnosis_id = save_diagnosis(
            image_path=f"/static/uploads/{filename}",
            disease_name=prediction['disease_name'],
            disease_name_tl=prediction['disease_name_tl'],
            confidence=prediction['confidence'],
            crop_type=prediction['crop_type'],
            gradcam_path=f"/static/uploads/{gradcam_filename}",
            treatment=treatment['en'],
            treatment_tl=treatment['tl']
        )

        return jsonify({
            'success': True,
            'diagnosis_id': diagnosis_id,
            'disease_name': prediction['disease_name'],
            'disease_name_tl': prediction['disease_name_tl'],
            'confidence': prediction['confidence'],
            'crop_type': prediction['crop_type'],
            'image_url': f"/static/uploads/{filename}",
            'gradcam_url': f"/static/uploads/{gradcam_filename}",
            'treatment': treatment[language] if language in treatment else treatment['en']
        })

    except Exception as e:
        current_app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Error sa pag-analyze ng larawan: {str(e)}'}), 500

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
        return "Diagnosis not found", 404
    return render_template('result.html', diagnosis=diagnosis)

@bp.route('/api/feedback/<int:diagnosis_id>', methods=['POST'])
def submit_feedback(diagnosis_id):
    """Submit user feedback on diagnosis accuracy"""
    data = request.get_json()
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
    """About page"""
    return render_template('about.html')
