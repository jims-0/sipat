"""
Sipat Routes
"""
import csv
import io
import json
import os
import uuid
import zipfile
from datetime import date, datetime, timedelta

from flask import (Blueprint, abort, current_app, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from PIL import Image, ImageOps, UnidentifiedImageError

from app.ml.adviser import ask
from app.ml.classes import CLASSES, CROPS, NOT_A_LEAF, OTHER_PLANT, SLUGS, is_diagnosis, slug
from app.ml.gradcam import generate_gradcam
from app.ml.inference import predict_disease
from app.ml.treatment import get_treatment_guidance
from app.models import (delete_diagnosis, delete_plot, get_dashboard, get_diagnosis_by_id,
                        get_diagnosis_history, get_due_rechecks, get_field_labelled,
                        get_followups, get_or_create_plot, get_plots, get_stats,
                        plot_exists, save_diagnosis, save_feedback, save_notes,
                        set_recheck)

bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
RECHECK_DAYS = (3, 7, 14)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def label_choices():
    """What the farmer can say a photo really was: the crop classes grouped by
    crop, then 'not a leaf' and 'another plant'."""
    groups = [(CROPS[crop], [(k, v) for k, v in CLASSES.items() if v['crop'] == crop])
              for crop in CROPS]
    groups.append(('Iba pa', [(NOT_A_LEAF, CLASSES[NOT_A_LEAF]), (OTHER_PLANT, CLASSES[OTHER_PLANT])]))
    return groups


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@bp.route('/')
def index():
    """Main page - camera/upload interface, plus re-checks that are due."""
    parent = get_diagnosis_by_id(_int_or_none(request.args.get('followup')) or 0)
    return render_template('index.html', plots=get_plots(), parent=parent,
                           due=get_due_rechecks(date.today().isoformat()), dash=get_dashboard())


@bp.route('/api/predict', methods=['POST'])
def predict():
    """Photo in, diagnosis out (saved to history)."""
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify({'error': 'Walang picture na na-upload'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Mali ang format ng picture. JPG o PNG lang.'}), 400

    # Our own name, never the uploaded one: no collisions, nothing user-controlled on disk.
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        with Image.open(file.stream) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            img.save(filepath, 'JPEG', quality=85, optimize=True)
    except (UnidentifiedImageError, OSError):
        return jsonify({'error': 'Hindi mabasa yung picture. Try ulit kumuha.'}), 400

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

    # A follow-up re-scan belongs to its first scan, and to the same plot
    # unless the farmer picked another one.
    parent = get_diagnosis_by_id(_int_or_none(request.form.get('parent_id')) or 0)
    plot_id = _int_or_none(request.form.get('plot_id'))
    if plot_id and not plot_exists(plot_id):
        plot_id = None
    plot_id = get_or_create_plot(request.form.get('new_plot')) or plot_id
    if parent and not plot_id:
        plot_id = parent['plot_id']

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
        plot_id=plot_id,
        parent_id=parent['id'] if parent else None,
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


@bp.route('/m/')
@bp.route('/m/<path:filename>')
def phone_app(filename='index.html'):
    """The phone app (pwa/), so a phone on the same Wi-Fi can open it from this
    PC during a demo. Over plain http it runs but cannot install for offline
    use - for that, host pwa/ on https (see pwa/README.md)."""
    if filename.startswith('_parity/'):
        abort(404)
    return send_from_directory(os.path.join(os.path.dirname(current_app.root_path), 'pwa'), filename)


@bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Photos live in UPLOAD_FOLDER, which in the desktop build is a folder next
    to the exe - never inside the bundle, which is deleted when the app closes."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@bp.route('/history')
def history():
    """Diagnosis history, filterable by plot, crop, result and a search."""
    filters = {
        'plot_id': _int_or_none(request.args.get('plot')),
        'crop': request.args.get('crop') or None,
        'status': request.args.get('status') or None,
        'query': (request.args.get('q') or '').strip()[:60] or None,
        'class_key': SLUGS.get(request.args.get('disease') or ''),
    }
    diagnoses = get_diagnosis_history(limit=100, **filters)
    return render_template('history.html', diagnoses=diagnoses, plots=get_plots(),
                           crops=CROPS, filters=filters)


def _diagnosis_or_404(diagnosis_id):
    diagnosis = get_diagnosis_by_id(diagnosis_id)
    if not diagnosis:
        abort(404)
    diagnosis['top3'] = json.loads(diagnosis.get('top3_json') or '[]')
    return diagnosis


@bp.route('/diagnosis/<int:diagnosis_id>')
def view_diagnosis(diagnosis_id):
    """View a single diagnosis in detail"""
    diagnosis = _diagnosis_or_404(diagnosis_id)
    parent = get_diagnosis_by_id(diagnosis['parent_id']) if diagnosis.get('parent_id') else None
    return render_template('result.html', diagnosis=diagnosis, parent=parent,
                           followups=get_followups(diagnosis_id), choices=label_choices(),
                           classes=CLASSES, recheck_days=RECHECK_DAYS,
                           library_slug=slug(diagnosis['class_key'])
                           if is_diagnosis(diagnosis.get('class_key') or '') else None)


@bp.route('/diagnosis/<int:diagnosis_id>/print')
def print_diagnosis(diagnosis_id):
    """One page to print and bring to the Municipal Agriculture Office."""
    diagnosis = _diagnosis_or_404(diagnosis_id)
    lang = 'en' if request.args.get('lang') == 'en' else 'tl'
    return render_template('print.html', diagnosis=diagnosis, lang=lang,
                           true_class=CLASSES.get(diagnosis.get('true_class_key') or ''),
                           printed=datetime.now())


@bp.route('/diagnosis/<int:diagnosis_id>/delete', methods=['POST'])
def remove_diagnosis(diagnosis_id):
    """Delete a scan and its photos."""
    row = delete_diagnosis(diagnosis_id)
    if not row:
        abort(404)
    folder = current_app.config['UPLOAD_FOLDER']
    for path in (row['image_path'], row['gradcam_path']):
        if path:
            # Only our own generated names, only inside the upload folder.
            target = os.path.join(folder, os.path.basename(path))
            if os.path.isfile(target):
                os.remove(target)
    return redirect(url_for('main.history'))


@bp.route('/api/ask/<int:diagnosis_id>', methods=['POST'])
def ask_adviser(diagnosis_id):
    """Follow-up question about one diagnosis, answered by the local adviser."""
    diagnosis = get_diagnosis_by_id(diagnosis_id)
    if not diagnosis:
        return jsonify({'error': 'Hindi makita yung scan'}), 404
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({'error': 'Walang tanong'}), 400
    return jsonify(ask(diagnosis, question, data.get('lang', 'tl')))


@bp.route('/api/feedback/<int:diagnosis_id>', methods=['POST'])
def submit_feedback(diagnosis_id):
    """Was it right - and, if the farmer knows, what was it really?

    {"helpful": 1} confirms a confident result. {"helpful": 0,
    "true_class_key": "..."} corrects it; {"true_class_key": "..."} alone
    labels an unsure / no-leaf result. These labels are the field test set."""
    diagnosis = get_diagnosis_by_id(diagnosis_id)
    if not diagnosis:
        return jsonify({'error': 'Hindi makita yung scan'}), 404
    data = request.get_json(silent=True) or {}
    true_key = data.get('true_class_key') or None
    if true_key is not None and true_key not in CLASSES:
        return jsonify({'error': 'Hindi kilalang sakit'}), 400
    helpful = data.get('helpful', None)
    if helpful is None and true_key is not None:
        helpful = true_key == diagnosis['class_key']
    if helpful is None:
        return jsonify({'error': 'Missing feedback value'}), 400

    save_feedback(diagnosis_id, bool(helpful), true_key)
    saved = get_diagnosis_by_id(diagnosis_id)
    return jsonify({'success': True, 'true_class_key': saved['true_class_key']})


@bp.route('/api/diagnosis/<int:diagnosis_id>/notes', methods=['POST'])
def update_notes(diagnosis_id):
    if not get_diagnosis_by_id(diagnosis_id):
        return jsonify({'error': 'Hindi makita yung scan'}), 404
    notes = ((request.get_json(silent=True) or {}).get('notes') or '').strip()[:1000]
    save_notes(diagnosis_id, notes)
    return jsonify({'success': True})


@bp.route('/api/diagnosis/<int:diagnosis_id>/recheck', methods=['POST'])
def update_recheck(diagnosis_id):
    """{"days": 7} sets a reminder to re-scan in a week; {"days": null} clears it."""
    if not get_diagnosis_by_id(diagnosis_id):
        return jsonify({'error': 'Hindi makita yung scan'}), 404
    days = (request.get_json(silent=True) or {}).get('days')
    if days is None:
        set_recheck(diagnosis_id, None)
        return jsonify({'success': True, 'recheck_due': None})
    days = _int_or_none(days)
    if days is None or not 1 <= days <= 60:
        return jsonify({'error': 'Pumili ng 1 to 60 days'}), 400
    due = (date.today() + timedelta(days=days)).isoformat()
    set_recheck(diagnosis_id, due)
    return jsonify({'success': True, 'recheck_due': due})


@bp.route('/plots', methods=['GET', 'POST'])
def plots():
    """Plots (taniman): each scan can be tagged to one, to follow a field over time."""
    if request.method == 'POST':
        get_or_create_plot(request.form.get('name'))
        return redirect(url_for('main.plots'))
    return render_template('plots.html', plots=get_plots())


@bp.route('/plots/<int:plot_id>/delete', methods=['POST'])
def remove_plot(plot_id):
    delete_plot(plot_id)
    return redirect(url_for('main.plots'))


@bp.route('/library')
def library():
    """Every disease Sipat knows, to read without taking a photo."""
    groups = [(crop, name_tl, [(slug(k), v) for k, v in CLASSES.items() if v['crop'] == crop])
              for crop, name_tl in CROPS.items()]
    return render_template('library.html', groups=groups, thumbs=_library_thumbs())


@bp.route('/library/<name>')
def library_entry(name):
    class_key = SLUGS.get(name)
    if not class_key or not is_diagnosis(class_key):
        abort(404)
    return render_template('library_entry.html', info=CLASSES[class_key], slug=name,
                           guidance=get_treatment_guidance(class_key),
                           thumb=name in _library_thumbs(),
                           scans=len(get_diagnosis_history(limit=1000, class_key=class_key)))


def _library_thumbs():
    folder = os.path.join(current_app.static_folder, 'library')
    return {f[:-4] for f in os.listdir(folder) if f.endswith('.jpg')} if os.path.isdir(folder) else set()


@bp.route('/export/field-set.zip')
def export_field_set():
    """The farmer-labelled photos, as <true class>/<photo>.jpg plus labels.csv -
    the input of scripts/evaluate_field.py."""
    rows = get_field_labelled()
    folder = current_app.config['UPLOAD_FOLDER']
    buf = io.BytesIO()
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(['id', 'file', 'true_class', 'model_class', 'model_status', 'confidence', 'timestamp'])
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            source = os.path.join(folder, os.path.basename(row['image_path']))
            if not os.path.isfile(source):
                continue
            arcname = f"{row['true_class_key']}/{os.path.basename(source)}"
            zf.write(source, arcname)
            writer.writerow([row['id'], arcname, row['true_class_key'], row['class_key'],
                             row['status'] or 'ok', row['confidence'], row['timestamp']])
        zf.writestr('labels.csv', table.getvalue())
    buf.seek(0)
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                     download_name=f"sipat_field_set_{date.today():%Y%m%d}.zip")


@bp.route('/stats')
def stats():
    """View usage statistics"""
    statistics = get_stats()
    from app.ml.inference import MODEL_DIR
    report_path = MODEL_DIR / 'field_report.json'
    field_report = json.loads(report_path.read_text()) if report_path.exists() else None
    return render_template('stats.html', stats=statistics, field_report=field_report)


@bp.route('/about')
def about():
    """About page. The measured numbers live on /training, not here."""
    return render_template('about.html')


# The training/evaluation files the evidence page reads and offers as downloads.
EVIDENCE_FILES = ('training_history.csv', 'training_info.json', 'evaluation_report.json')


def _accuracy_chart(history, width=600, height=240, pad=36, low=0.5):
    """SVG coordinates for train/val accuracy per epoch (y axis from `low` to 1.0)."""
    last = max(len(history) - 1, 1)

    def point(i, value):
        x = pad + (width - 2 * pad) * i / last
        y = pad + (height - 2 * pad) * (1 - (max(value, low) - low) / (1 - low))
        return round(x, 1), round(y, 1)

    def line(key):
        return ' '.join(f'{x},{y}' for x, y in (point(i, row[key]) for i, row in enumerate(history)))

    ticks = [(round(v, 1), point(0, v)[1]) for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    epochs = [(i + 1, point(i, low)[0]) for i in range(len(history))]
    return {'width': width, 'height': height, 'pad': pad, 'train': line('accuracy'),
            'val': line('val_accuracy'), 'ticks': ticks, 'epochs': epochs, 'point': point}


@bp.route('/training')
def training():
    """Evidence of how the model was trained and tested, for the thesis panel:
    everything here is read from the files training/evaluation wrote."""
    from app.ml.inference import MODEL_DIR
    paths = {name: MODEL_DIR / name for name in EVIDENCE_FILES}
    info = json.loads(paths['training_info.json'].read_text()) \
        if paths['training_info.json'].exists() else None
    report = json.loads(paths['evaluation_report.json'].read_text()) \
        if paths['evaluation_report.json'].exists() else None
    history = []
    if paths['training_history.csv'].exists():
        with open(paths['training_history.csv'], newline='', encoding='utf-8') as handle:
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]

    chart = _accuracy_chart(history) if len(history) > 1 else None
    if chart and info and info.get('head_epochs', 0) < len(history):
        # The head-only epochs come first; fine-tuning starts right after them.
        chart['fine_x'] = chart['point'](info['head_epochs'], 0.5)[0]
    names = {row['class']: row['name'] for row in report['per_class']} if report else {}
    matrix = None
    if report and report.get('confusion_matrix'):
        matrix = [{'name': row['name'], 'total': sum(cells),
                   'cells': [(n, n / max(sum(cells), 1)) for n in cells]}
                  for row, cells in zip(report['per_class'], report['confusion_matrix'])]
    return render_template('training.html', info=info, report=report, history=history,
                           chart=chart, names=names, matrix=matrix,
                           files=[name for name, path in paths.items() if path.exists()])


@bp.route('/training/<name>')
def training_file(name):
    if name not in EVIDENCE_FILES:
        abort(404)
    from app.ml.inference import MODEL_DIR
    return send_from_directory(MODEL_DIR, name, as_attachment=True)
