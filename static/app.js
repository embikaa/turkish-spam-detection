// API Base URL
const API_URL = 'http://localhost:8000';

// Chart instances
let metricsChart, confusionChart, rocChart, classChart;

// Load model info on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadModelInfo();
    initializeCharts();
});

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_URL}/model-info`);
        const data = await response.json();

        // Update model info banner
        document.getElementById('modelVersion').textContent =
            data.version ? new Date(data.version).toLocaleString('tr-TR') : 'N/A';

        if (data.metrics) {
            const metrics = data.metrics;

            // Update banner
            document.getElementById('modelAccuracy').textContent =
                (metrics.accuracy * 100).toFixed(2) + '%';
            document.getElementById('modelF1').textContent =
                (metrics.f1_score * 100).toFixed(2) + '%';
            document.getElementById('modelPrecision').textContent =
                (metrics.precision * 100).toFixed(2) + '%';
            document.getElementById('modelRecall').textContent =
                (metrics.recall * 100).toFixed(2) + '%';

            // Update metrics cards
            document.getElementById('metricAccuracy').textContent =
                (metrics.accuracy * 100).toFixed(1) + '%';
            document.getElementById('metricF1').textContent =
                (metrics.f1_score * 100).toFixed(1) + '%';
            document.getElementById('metricPrecision').textContent =
                (metrics.precision * 100).toFixed(1) + '%';
            document.getElementById('metricRecall').textContent =
                (metrics.recall * 100).toFixed(1) + '%';

            // Update charts
            updateMetricsChart(metrics);
            updateConfusionMatrix(metrics.confusion_matrix);
            updateClassDistribution(metrics.class_distribution);

            if (metrics.roc_auc) {
                document.getElementById('aucBadge').textContent =
                    `AUC: ${metrics.roc_auc.toFixed(3)}`;
                updateROCCurve(metrics.roc_auc);
            }
        }
    } catch (error) {
        console.error('Model bilgisi yüklenemedi:', error);
        document.getElementById('modelVersion').textContent = 'Hata';
    }
}

// Initialize all charts
function initializeCharts() {
    // Metrics bar chart
    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(metricsCtx, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
            datasets: [{
                label: 'Performans Metrikleri',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(168, 85, 247, 0.8)'
                ],
                borderColor: [
                    'rgb(99, 102, 241)',
                    'rgb(16, 185, 129)',
                    'rgb(245, 158, 11)',
                    'rgb(168, 85, 247)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#94a3b8',
                        callback: value => value + '%'
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                },
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Confusion Matrix
    const confusionCtx = document.getElementById('confusionMatrix').getContext('2d');
    confusionChart = new Chart(confusionCtx, {
        type: 'bar',
        data: {
            labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
            datasets: [{
                label: 'Sayı',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(16, 185, 129, 0.8)'
                ],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                x: {
                    ticks: { color: '#94a3b8', font: { size: 10 } },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // ROC Curve (placeholder)
    const rocCtx = document.getElementById('rocCurve').getContext('2d');
    rocChart = new Chart(rocCtx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 11 }, (_, i) => (i / 10).toFixed(1)),
            datasets: [{
                label: 'ROC Curve',
                data: Array.from({ length: 11 }, (_, i) => i / 10),
                borderColor: 'rgb(99, 102, 241)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3
            }, {
                label: 'Random Classifier',
                data: Array.from({ length: 11 }, (_, i) => i / 10),
                borderColor: 'rgba(148, 163, 184, 0.5)',
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    title: { display: true, text: 'True Positive Rate', color: '#94a3b8' },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                x: {
                    title: { display: true, text: 'False Positive Rate', color: '#94a3b8' },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });

    // Class Distribution
    const classCtx = document.getElementById('classDistribution').getContext('2d');
    classChart = new Chart(classCtx, {
        type: 'doughnut',
        data: {
            labels: ['Genuine', 'Spam'],
            datasets: [{
                data: [0, 0],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgb(16, 185, 129)',
                    'rgb(239, 68, 68)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { size: 14 } }
                }
            }
        }
    });
}

// Update charts with data
function updateMetricsChart(metrics) {
    metricsChart.data.datasets[0].data = [
        metrics.accuracy * 100,
        metrics.f1_score * 100,
        metrics.precision * 100,
        metrics.recall * 100
    ];
    metricsChart.update();
}

function updateConfusionMatrix(matrix) {
    if (!matrix || matrix.length !== 2) return;

    // matrix = [[TN, FP], [FN, TP]]
    confusionChart.data.datasets[0].data = [
        matrix[0][0], // TN
        matrix[0][1], // FP
        matrix[1][0], // FN
        matrix[1][1]  // TP
    ];
    confusionChart.update();
}

function updateROCCurve(auc) {
    // Generate approximate ROC curve based on AUC
    const points = 11;
    const data = Array.from({ length: points }, (_, i) => {
        const x = i / (points - 1);
        return Math.min(1, x + (auc - 0.5) * 2 * (1 - x));
    });

    rocChart.data.datasets[0].data = data;
    rocChart.update();
}

function updateClassDistribution(distribution) {
    if (!distribution) return;

    const genuine = distribution['Genuine'] || 0;
    const spam = distribution['Spam'] || 0;

    classChart.data.datasets[0].data = [genuine, spam];
    classChart.update();
}

// Handle prediction form
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const text = document.getElementById('reviewText').value.trim();
    if (!text) return;

    // Show loading
    document.getElementById('btnText').style.display = 'none';
    document.getElementById('btnLoading').style.display = 'inline-block';
    document.getElementById('predictBtn').disabled = true;

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const result = await response.json();
        displayResult(result);
    } catch (error) {
        console.error('Tahmin hatası:', error);
        alert('Tahmin yapılırken bir hata oluştu. API çalışıyor mu?');
    } finally {
        // Hide loading
        document.getElementById('btnText').style.display = 'inline';
        document.getElementById('btnLoading').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
    }
});

// Display prediction result
function displayResult(result) {
    const container = document.getElementById('resultContainer');
    const isSpam = result.is_spam;

    container.className = `result ${isSpam ? 'spam' : 'genuine'}`;
    container.innerHTML = `
        <div class="result-label">
            ${isSpam ? '🔴 SPAM' : '🟢 GERÇEK (GENUINE)'}
        </div>
        <div class="result-details">
            <div class="detail-item">
                <span>Spam İhtimali:</span>
                <strong>${(result.spam_probability * 100).toFixed(2)}%</strong>
            </div>
            <div class="detail-item">
                <span>Güven Seviyesi:</span>
                <strong>${result.confidence}</strong>
            </div>
        </div>
    `;
    container.style.display = 'block';
}
