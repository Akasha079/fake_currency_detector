const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resultContainer = document.getElementById('result-container');
const resetBtn = document.getElementById('reset-btn');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const predictionLabel = document.getElementById('prediction-label');

let currentFile = null;

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.style.display = 'none';
        previewContainer.style.display = 'block';
        resultContainer.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    analyzeBtn.textContent = "Analyzing...";
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        showResult(data);
    } catch (error) {
        alert('Error analyzing image. Please try again.');
        console.error(error);
    } finally {
        analyzeBtn.textContent = "Analyze Currency";
        analyzeBtn.disabled = false;
    }
});

function showResult(data) {
    previewContainer.style.display = 'none';
    resultContainer.style.display = 'block';

    predictionLabel.textContent = data.label;
    predictionLabel.className = 'label ' + (data.label.toLowerCase().includes('real') ? 'real' : 'fake');

    // Animate bar
    setTimeout(() => {
        const percent = Math.round(data.confidence * 100);
        confidenceBar.style.width = `${percent}%`;
        confidenceText.textContent = `Confidence: ${percent}%`;
    }, 100);
}

resetBtn.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    resultContainer.style.display = 'none';
    dropZone.style.display = 'block';
    confidenceBar.style.width = '0%';
});
