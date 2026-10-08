/**
 * Tanim Guard - Camera & Upload Handler
 */

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const cameraInput = document.getElementById('camera-input');
    const previewImg = document.getElementById('preview-img');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const currentLang = document.documentElement.lang || 'tl';

    let selectedFile = null;

    // Handle file selection
    function handleFile(file) {
        if (!file) return;

        // Validate type
        if (!file.type.match('image.*')) {
            alert(currentLang === 'tl' ? 'Pumili ng larawan lamang.' : 'Please select an image file.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            if (previewPlaceholder) {
                previewPlaceholder.style.display = 'none';
            }
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
    }

    if (cameraInput) {
        cameraInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
    }

    // Handle analysis submit
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            // Show loading
            loadingOverlay.classList.add('active');

            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('language', currentLang);

            try {
                // Get CSRF token
                const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': csrfToken || ''
                    },
                    body: formData
                });

                const result = await response.json();

                if (response.ok && result.success) {
                    // Redirect to diagnosis result view
                    window.location.href = `/diagnosis/${result.diagnosis_id}`;
                } else {
                    alert(result.error || (currentLang === 'tl' ? 'May naganap na error.' : 'An error occurred.'));
                    loadingOverlay.classList.remove('active');
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert(currentLang === 'tl' ? 'Hindi ma-connect sa server.' : 'Cannot connect to server.');
                loadingOverlay.classList.remove('active');
            }
        });
    }
});
