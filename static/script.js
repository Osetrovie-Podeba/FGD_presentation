// Enhanced JavaScript with dataset selection
class FishGenderApp {
    constructor() {
        this.selectedImages = new Set();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.setupDatasetSelection();
    }

    setupEventListeners() {
        const fileInput = document.getElementById('fileInput');
        fileInput.addEventListener('change', (e) => this.handleFiles(e.target.files));
    }

    setupDragAndDrop() {
        const uploadZone = document.getElementById('uploadZone');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
        });

        uploadZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFiles(files);
        }, false);
    }

    setupDatasetSelection() {
        const datasetCards = document.querySelectorAll('.dataset-card');
        datasetCards.forEach(card => {
            card.addEventListener('click', () => this.toggleImageSelection(card));
        });
    }

    toggleImageSelection(card) {
        const imageId = card.dataset.id;

        if (this.selectedImages.has(imageId)) {
            this.selectedImages.delete(imageId);
            card.classList.remove('selected');
        } else {
            this.selectedImages.add(imageId);
            card.classList.add('selected');
        }

        this.updateSelectionUI();
    }

    updateSelectionUI() {
        const selectedCount = this.selectedImages.size;
        const countElement = document.getElementById('selectedCount');
        const analyzeBtn = document.querySelector('.analyze-btn');

        countElement.textContent = selectedCount;
        analyzeBtn.disabled = selectedCount === 0;

        if (selectedCount > 0) {
            analyzeBtn.style.background = 'linear-gradient(135deg, #4CAF50, #8BC34A)';
        } else {
            analyzeBtn.style.background = '#ccc';
        }
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    async handleFiles(files) {
        if (files.length === 0) return;

        this.showLoading();

        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            this.displayResults(data.results, data.demo_mode);
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing images. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async analyzeSelected() {
        if (this.selectedImages.size === 0) return;

        this.showLoading();

        try {
            const response = await fetch('/predict-selected', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(Array.from(this.selectedImages))
            });

            const data = await response.json();
            this.displayResults(data.results, data.demo_mode);
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing selected images. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        const progressFill = document.getElementById('progressFill');

        overlay.style.display = 'flex';

        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
        }, 200);

        overlay.dataset.interval = interval;
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        const progressFill = document.getElementById('progressFill');

        progressFill.style.width = '100%';

        setTimeout(() => {
            overlay.style.display = 'none';
            progressFill.style.width = '0%';

            if (overlay.dataset.interval) {
                clearInterval(overlay.dataset.interval);
            }
        }, 500);
    }

    displayResults(results, demoMode) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsGrid = document.getElementById('resultsGrid');
        const statsContainer = document.getElementById('statsContainer');

        resultsSection.style.display = 'block';
        statsContainer.style.display = 'grid';
        resultsGrid.innerHTML = '';

        // Show demo mode banner if active
        if (demoMode) {
            const demoBanner = document.createElement('div');
            demoBanner.className = 'demo-banner';
            demoBanner.innerHTML = '🎭 DEMO MODE: Using simulated predictions for demonstration';
            resultsGrid.appendChild(demoBanner);
        }

        let maleCount = 0;
        let femaleCount = 0;
        let totalConfidence = 0;
        let validResults = 0;

        results.forEach((result, index) => {
            if (result.error) {
                this.createErrorCard(result, resultsGrid);
                return;
            }

            const card = this.createResultCard(result, index);
            resultsGrid.appendChild(card);

            if (result.gender === 'Male') maleCount++;
            else femaleCount++;

            totalConfidence += result.confidence;
            validResults++;
        });

        this.updateStatistics(maleCount, femaleCount, totalConfidence, validResults);
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    createResultCard(result, index) {
        const card = document.createElement('div');
        card.className = `result-card ${result.is_predefined ? 'predefined' : ''}`;
        card.style.animationDelay = `${index * 0.1}s`;

        const genderClass = result.gender.toLowerCase();

        let comparisonHTML = '';
        if (result.is_predefined && result.expected_gender) {
            const isCorrect = result.gender === result.expected_gender;
            comparisonHTML = `
                <div class="result-comparison">
                    <h5>📊 Prediction Analysis</h5>
                    <div class="comparison-item">
                        <span>Expected:</span>
                        <span class="expected-gender gender-${result.expected_gender.toLowerCase()}">
                            ${result.expected_gender}
                        </span>
                    </div>
                    <div class="comparison-item">
                        <span>Predicted:</span>
                        <span class="accuracy-indicator accuracy-${isCorrect ? 'correct' : 'incorrect'}">
                            ${isCorrect ? '✓ Correct' : '✗ Different'}
                        </span>
                    </div>
                </div>
            `;
        }

        card.innerHTML = `
            <img src="${result.image_path}" alt="${result.filename}" class="result-image">
            <div class="result-info">
                <h3>${result.filename}</h3>
                ${result.description ? `<p class="image-description">${result.description}</p>` : ''}
                <div class="gender-badge gender-${genderClass}">
                    <i class="fas fa-${result.gender === 'Male' ? 'mars' : 'venus'}"></i>
                    ${result.gender}
                </div>
                <p><strong>Confidence:</strong> ${result.confidence}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence}%"></div>
                </div>
                ${comparisonHTML}
            </div>
        `;

        return card;
    }

    createErrorCard(result, container) {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.style.borderLeft = '5px solid #f44336';

        card.innerHTML = `
            <div class="result-info">
                <h3>${result.filename}</h3>
                <p style="color: #f44336;"><strong>Error:</strong> ${result.error}</p>
            </div>
        `;

        container.appendChild(card);
    }

    updateStatistics(maleCount, femaleCount, totalConfidence, validResults) {
        document.getElementById('maleCount').textContent = maleCount;
        document.getElementById('femaleCount').textContent = femaleCount;

        const avgConfidence = validResults > 0 ? (totalConfidence / validResults).toFixed(1) : 0;
        document.getElementById('avgConfidence').textContent = avgConfidence + '%';

        this.animateCounters();
    }

    animateCounters() {
        const counters = document.querySelectorAll('.stat-value');
        counters.forEach(counter => {
            const target = parseInt(counter.textContent) || parseFloat(counter.textContent) || 0;
            const increment = target / 20;
            let current = 0;

            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    counter.textContent = counter.id === 'avgConfidence' ? target + '%' : Math.ceil(target);
                    clearInterval(timer);
                } else {
                    counter.textContent = counter.id === 'avgConfidence' ?
                        Math.floor(current) + '%' : Math.floor(current);
                }
            }, 50);
        });
    }
}

// Global functions for HTML onclick handlers
function switchTab(tabName) {
    const tabs = document.querySelectorAll('.tab-btn');
    const predefinedSection = document.getElementById('predefinedSection');
    const uploadSection = document.getElementById('uploadSection');

    tabs.forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');

    if (tabName === 'predefined') {
        predefinedSection.style.display = 'block';
        uploadSection.style.display = 'none';
    } else {
        predefinedSection.style.display = 'none';
        uploadSection.style.display = 'block';
    }
}

function selectAll() {
    const app = window.fishApp;
    const datasetCards = document.querySelectorAll('.dataset-card');

    datasetCards.forEach(card => {
        const imageId = card.dataset.id;
        app.selectedImages.add(imageId);
        card.classList.add('selected');
    });

    app.updateSelectionUI();
}

function clearSelection() {
    const app = window.fishApp;
    const datasetCards = document.querySelectorAll('.dataset-card');

    app.selectedImages.clear();
    datasetCards.forEach(card => {
        card.classList.remove('selected');
    });

    app.updateSelectionUI();
}

function analyzeSelected() {
    window.fishApp.analyzeSelected();
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.fishApp = new FishGenderApp();
});
