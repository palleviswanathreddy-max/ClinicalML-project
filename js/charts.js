// ─── Chart Manager ───────────────────────────────────────────────────────────

const ChartManager = {
  charts: {},

  destroy(id) {
    if (this.charts[id]) {
      this.charts[id].destroy();
      delete this.charts[id];
    }
  },

  // ── EDA: Correlation Heatmap ─────────────────────────────────────────────
  renderCorrelationHeatmap(canvasId) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Use real computed correlation matrix if available, otherwise fallback
    const labels = (typeof COMPUTED_CORRELATION !== 'undefined') ? COMPUTED_CORRELATION.labels
      : ['BMI', 'Glucose', 'LDL', 'HDL', 'SBP', 'DBP', 'HR', 'Age'];
    const matrix = (typeof COMPUTED_CORRELATION !== 'undefined') ? COMPUTED_CORRELATION.matrix
      : [
        [1.00,  0.62,  0.45, -0.38,  0.51,  0.48,  0.21,  0.34],
        [0.62,  1.00,  0.38, -0.42,  0.44,  0.40,  0.19,  0.47],
        [0.45,  0.38,  1.00, -0.65,  0.39,  0.33,  0.14,  0.38],
        [-0.38,-0.42, -0.65,  1.00, -0.35, -0.30, -0.12, -0.28],
        [0.51,  0.44,  0.39, -0.35,  1.00,  0.72,  0.35,  0.52],
        [0.48,  0.40,  0.33, -0.30,  0.72,  1.00,  0.30,  0.46],
        [0.21,  0.19,  0.14, -0.12,  0.35,  0.30,  1.00,  0.22],
        [0.34,  0.47,  0.38, -0.28,  0.52,  0.46,  0.22,  1.00]
      ];

    // Build cell data for custom draw
    const n = labels.length;
    const size = Math.min(canvas.offsetWidth || 400, 400);
    canvas.width = size;
    canvas.height = size;
    const cellW = size / n;
    const cellH = size / n;

    const getColor = (val) => {
      if (val >= 0) {
        const intensity = Math.round(val * 255);
        return `rgb(${255 - intensity}, ${100 + Math.round(val * 100)}, ${255 - intensity})`;
      } else {
        const intensity = Math.round(-val * 255);
        return `rgb(${255}, ${100 - Math.round(-val * 60)}, ${100 - Math.round(-val * 60)})`;
      }
    };

    ctx.clearRect(0, 0, size, size);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const val = matrix[r][c];
        ctx.fillStyle = getColor(val);
        ctx.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);

        // Label
        ctx.fillStyle = Math.abs(val) > 0.6 ? '#fff' : '#1e293b';
        ctx.font = `bold ${Math.round(cellW * 0.22)}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(2), c * cellW + cellW / 2, r * cellH + cellH / 2);
      }
    }

    // Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = `${Math.round(cellW * 0.22)}px Inter, sans-serif`;
    // Store labels on canvas for axis rendering in CSS overlay
    canvas.dataset.labels = JSON.stringify(labels);
  },

  // ── EDA: Feature Distribution ───────────────────────────────────────────
  renderDistributions(canvasId, patientData) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Generate synthetic population + patient overlay
    const generateNormal = (mean, sd, n) =>
      Array.from({ length: n }, () => {
        const u1 = Math.random(), u2 = Math.random();
        return mean + sd * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      });

    const bmiPop = generateNormal(27.5, 5.2, 300).filter(v => v > 15 && v < 50);
    const bins = (data, min, max, n) => {
      const w = (max - min) / n;
      const counts = new Array(n).fill(0);
      data.forEach(v => {
        const i = Math.floor((v - min) / w);
        if (i >= 0 && i < n) counts[i]++;
      });
      return { counts, labels: Array.from({ length: n }, (_, i) => (min + i * w + w / 2).toFixed(1)) };
    };

    const { counts, labels } = bins(bmiPop, 15, 50, 14);

    this.charts[canvasId] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Population Distribution',
            data: counts,
            backgroundColor: 'rgba(124, 58, 237, 0.4)',
            borderColor: 'rgba(124, 58, 237, 0.8)',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8' } },
          title: {
            display: true,
            text: 'BMI Distribution — Population vs Patient',
            color: '#e2e8f0',
            font: { size: 14, weight: 'bold' }
          },
          annotation: {
            annotations: {
              patientLine: {
                type: 'line',
                xMin: patientData?.bmi || 27,
                xMax: patientData?.bmi || 27,
                borderColor: '#06b6d4',
                borderWidth: 2,
                label: { content: 'You', display: true }
              }
            }
          }
        },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  },

  // ── Patient Radar vs Population ─────────────────────────────────────────
  renderRadar(canvasId, patientData) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Hide the placeholder hint once real data is available
    const hint = document.getElementById('radar-hint');
    if (hint) hint.style.display = 'none';

    const normalize100 = (val, min, max) =>
      Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));

    const patientValues = [
      normalize100(patientData.bmi || 25, 15, 50),
      normalize100(patientData.glucose || 90, 60, 300),
      normalize100(patientData.ldl || 120, 40, 250),
      normalize100(100 - (patientData.hdl || 55), 0, 100),
      normalize100(patientData.systolic || 120, 80, 200),
      normalize100(10 - (patientData.exercise_hours || 3), 0, 10),
      normalize100(patientData.age || 45, 18, 90)
    ];

    const popNorms = [
      normalize100(27.5, 15, 50),
      normalize100(99, 60, 300),
      normalize100(130, 40, 250),
      normalize100(100 - 52, 0, 100),
      normalize100(124, 80, 200),
      normalize100(10 - 3.5, 0, 10),
      normalize100(48, 18, 90)
    ];

    this.charts[canvasId] = new Chart(canvas, {
      type: 'radar',
      data: {
        labels: ['BMI Risk', 'Glucose Risk', 'LDL Risk', 'Low HDL Risk', 'BP Risk', 'Inactivity Risk', 'Age Risk'],
        datasets: [
          {
            label: 'This Patient',
            data: patientValues,
            backgroundColor: 'rgba(6, 182, 212, 0.15)',
            borderColor: '#06b6d4',
            borderWidth: 2,
            pointBackgroundColor: '#06b6d4',
            pointRadius: 4
          },
          {
            label: 'Population Average',
            data: popNorms,
            backgroundColor: 'rgba(124, 58, 237, 0.10)',
            borderColor: 'rgba(124, 58, 237, 0.6)',
            borderWidth: 2,
            pointBackgroundColor: 'rgba(124, 58, 237, 0.8)',
            pointRadius: 3,
            borderDash: [5, 3]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { size: 12 } } }
        },
        scales: {
          r: {
            min: 0, max: 100,
            ticks: { display: false },
            grid: { color: 'rgba(255,255,255,0.08)' },
            angleLines: { color: 'rgba(255,255,255,0.08)' },
            pointLabels: { color: '#94a3b8', font: { size: 11 } }
          }
        }
      }
    });
  },

  // ── Disease Classification Probability Bars ──────────────────────────────
  renderClassificationChart(canvasId, results) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const labels = DISEASE_CLASSES.map(d => d.label);
    const dnnProbs = results.stage1.dnn.probabilities.map(p => (p * 100).toFixed(1));
    const rfProbs = results.stage1.random_forest.probabilities.map(p => (p * 100).toFixed(1));
    const gbProbs = results.stage1.gradient_boosting.probabilities.map(p => (p * 100).toFixed(1));
    const ensProbs = results.stage1.ensemble.probabilities.map(p => (p * 100).toFixed(1));

    this.charts[canvasId] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'DNN',
            data: dnnProbs,
            backgroundColor: 'rgba(124, 58, 237, 0.7)',
            borderRadius: 4
          },
          {
            label: 'Random Forest',
            data: rfProbs,
            backgroundColor: 'rgba(6, 182, 212, 0.7)',
            borderRadius: 4
          },
          {
            label: 'Gradient Boosting',
            data: gbProbs,
            backgroundColor: 'rgba(245, 158, 11, 0.7)',
            borderRadius: 4
          },
          {
            label: 'Ensemble',
            data: ensProbs,
            backgroundColor: 'rgba(16, 185, 129, 0.7)',
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8' } },
          title: {
            display: true,
            text: 'Model Confidence by Disease Class (%)',
            color: '#e2e8f0',
            font: { size: 14, weight: 'bold' }
          }
        },
        scales: {
          x: {
            ticks: { color: '#94a3b8', maxRotation: 30 },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            max: 100,
            ticks: { color: '#94a3b8', callback: v => v + '%' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          }
        }
      }
    });
  },

  // ── Feature Importance Chart ─────────────────────────────────────────────
  renderFeatureImportance(canvasId) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const featureData = (typeof COMPUTED_FEATURE_IMPORTANCE !== 'undefined') ? COMPUTED_FEATURE_IMPORTANCE : FEATURE_IMPORTANCE;
    const sorted = [...featureData].sort((a, b) => b.importance - a.importance);
    const categoryColors = {
      Physiological: 'rgba(124, 58, 237, 0.75)',
      Demographic: 'rgba(6, 182, 212, 0.75)',
      Demographics: 'rgba(6, 182, 212, 0.75)',
      Lifestyle: 'rgba(245, 158, 11, 0.75)',
      'Medical History': 'rgba(16, 185, 129, 0.75)'
    };

    this.charts[canvasId] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: sorted.map(f => f.feature),
        datasets: [{
          label: 'Feature Importance Score',
          data: sorted.map(f => (f.importance * 100).toFixed(2)),
          backgroundColor: sorted.map(f => categoryColors[f.category] || '#7c3aed'),
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'Feature Importance — Drug Recommendation Model (%)',
            color: '#e2e8f0',
            font: { size: 14, weight: 'bold' }
          },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.parsed.x}% importance`
            }
          }
        },
        scales: {
          x: {
            ticks: { color: '#94a3b8', callback: v => v + '%' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            ticks: { color: '#94a3b8', font: { size: 11 } },
            grid: { display: false }
          }
        }
      }
    });
  },

  // ── Pie: Diet Confidence Distribution ───────────────────────────────────
  renderDietPie(canvasId, dietResults) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const colors = ['#7c3aed', '#06b6d4', '#f59e0b', '#ef4444', '#10b981', '#64748b'];

    this.charts[canvasId] = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels: dietResults.all_scores.map(s => s.label),
        datasets: [{
          data: dietResults.all_scores.map(s => (s.confidence * 100).toFixed(1)),
          backgroundColor: colors,
          borderColor: 'transparent',
          hoverOffset: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '60%',
        plugins: {
          legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } }
        }
      }
    });
  },

  // ── Doughnut: Drug Confidence Distribution ───────────────────────────────
  renderDrugPie(canvasId, drugResults) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const colors = ['#ef4444', '#8b5cf6', '#f59e0b', '#dc2626', '#10b981', '#06b6d4'];

    this.charts[canvasId] = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels: drugResults.all_scores.map(s => s.label),
        datasets: [{
          data: drugResults.all_scores.map(s => (s.confidence * 100).toFixed(1)),
          backgroundColor: colors,
          borderColor: 'transparent',
          hoverOffset: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } }
        }
      }
    });
  }
};
