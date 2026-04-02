// ─── UI Manager ──────────────────────────────────────────────────────────────

let currentStep = 1;
const TOTAL_STEPS = 4;
let lastResults = null;
let patientData = {};

// ── Step Navigation ──────────────────────────────────────────────────────────
function goToStep(step) {
  if (step < 1 || step > TOTAL_STEPS) return;

  // Validate current step before advancing
  if (step > currentStep && !validateStep(currentStep)) return;

  document.querySelectorAll('.form-step').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.step-indicator').forEach((el, i) => {
    el.classList.toggle('active', i + 1 === step);
    el.classList.toggle('completed', i + 1 < step);
  });

  const stepEl = document.getElementById(`step-${step}`);
  if (stepEl) {
    stepEl.classList.add('active');
    stepEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  currentStep = step;
  updateNavButtons();
}

function updateNavButtons() {
  const prevBtn = document.getElementById('prev-btn');
  const nextBtn = document.getElementById('next-btn');
  const submitBtn = document.getElementById('submit-btn');

  if (prevBtn) prevBtn.style.display = currentStep === 1 ? 'none' : 'flex';
  if (nextBtn) nextBtn.style.display = currentStep === TOTAL_STEPS ? 'none' : 'flex';
  if (submitBtn) submitBtn.style.display = currentStep === TOTAL_STEPS ? 'flex' : 'none';
}

// ── Form Validation ──────────────────────────────────────────────────────────
function validateStep(step) {
  const required = {
    1: ['age', 'gender'],
    2: [],
    3: ['bmi', 'glucose', 'ldl', 'hdl', 'systolic', 'diastolic'],
    4: ['activity_level']
  };

  const fields = required[step] || [];
  let valid = true;

  fields.forEach(fieldId => {
    const el = document.getElementById(fieldId);
    if (!el || !el.value) {
      el?.classList.add('error');
      valid = false;
    } else {
      el?.classList.remove('error');
    }
  });

  if (!valid) {
    showToast('Please fill in all required fields.', 'error');
  }
  return valid;
}

// ── Collect Form Data ────────────────────────────────────────────────────────
function collectFormData() {
  const get = id => document.getElementById(id)?.value;
  const getNum = id => parseFloat(get(id)) || 0;
  const getChecked = id => document.getElementById(id)?.checked;

  return {
    // Demographics
    age: getNum('age'),
    gender: get('gender'),
    ethnicity: get('ethnicity'),

    // Medical History
    family_history: getChecked('family_history'),
    history_diabetes: getChecked('history_diabetes'),
    history_hypertension: getChecked('history_hypertension'),
    history_cvd: getChecked('history_cvd'),
    // FIX: these 4 fields were shown in the form but never collected
    history_metabolic: getChecked('history_metabolic'),
    history_kidney: getChecked('history_kidney'),
    current_meds: get('current_meds') || 'none',
    allergies: get('allergies') || 'none',

    // Physiological
    bmi: getNum('bmi'),
    glucose: getNum('glucose'),
    ldl: getNum('ldl'),
    hdl: getNum('hdl'),
    systolic: getNum('systolic'),
    diastolic: getNum('diastolic'),
    heart_rate: getNum('heart_rate'),

    // Lifestyle
    activity_level: getNum('activity_level'),
    exercise_hours: getNum('exercise_hours'),
    smoking: get('smoking'),
    alcohol: get('alcohol')
  };
}

// ── Sliders with Live Value Display ─────────────────────────────────────────
function initSliders() {
  document.querySelectorAll('input[type="range"]').forEach(slider => {
    const display = document.getElementById(`${slider.id}-val`);
    if (display) display.textContent = slider.value;

    slider.addEventListener('input', () => {
      if (display) display.textContent = slider.value;
      slider.style.background = getSliderGradient(slider);
    });
    slider.style.background = getSliderGradient(slider);
  });
}

function getSliderGradient(slider) {
  const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
  return `linear-gradient(to right, #7c3aed ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
}

// ── Toast Notifications ──────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${type === 'error' ? '⚠️' : type === 'success' ? '✅' : 'ℹ️'}</span>
    <span>${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => toast.classList.add('show'), 10);
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ── Section Navigation (Header Tabs) ────────────────────────────────────────
function showSection(sectionId) {
  document.querySelectorAll('.app-section').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => {
    el.classList.toggle('active', el.dataset.section === sectionId);
  });
  const section = document.getElementById(sectionId);
  if (section) {
    section.classList.add('active');
    // Lazy-render EDA & feature importance when those tabs open
    if (sectionId === 'eda-section') renderEDA();
    if (sectionId === 'importance-section') renderImportance();
  }
}

// ── Progress Loading Overlay ─────────────────────────────────────────────────
function showLoading() {
  document.getElementById('loading-overlay')?.classList.add('active');
}
function hideLoading() {
  document.getElementById('loading-overlay')?.classList.remove('active');
}

// ── EDA Section ──────────────────────────────────────────────────────────────
function renderEDA() {
  ChartManager.renderCorrelationHeatmap('correlation-canvas');
  ChartManager.renderDistributions('distribution-canvas', patientData);
  if (lastResults) {
    ChartManager.renderRadar('radar-canvas', lastResults.patient);
  }
}

// ── Feature Importance Section ───────────────────────────────────────────────
function renderImportance() {
  ChartManager.renderFeatureImportance('importance-canvas');

  // Render ranked list
  const listEl = document.getElementById('importance-list');
  if (!listEl) return;

  const featureData = (typeof COMPUTED_FEATURE_IMPORTANCE !== 'undefined') ? COMPUTED_FEATURE_IMPORTANCE : FEATURE_IMPORTANCE;
  const sorted = [...featureData].sort((a, b) => b.importance - a.importance);
  const categoryColors = {
    Physiological: '#7c3aed',
    Demographic: '#06b6d4',
    Demographics: '#06b6d4',
    Lifestyle: '#f59e0b',
    'Medical History': '#10b981'
  };

  listEl.innerHTML = sorted.map((f, i) => `
    <div class="importance-item">
      <div class="importance-rank">${i + 1}</div>
      <div class="importance-info">
        <div class="importance-name">${f.feature}</div>
        <div class="importance-bar-wrap">
          <div class="importance-bar" style="width: ${(f.importance / 0.187 * 100).toFixed(0)}%; background: ${categoryColors[f.category]}"></div>
        </div>
      </div>
      <div class="importance-score" style="color: ${categoryColors[f.category]}">
        ${(f.importance * 100).toFixed(1)}%
      </div>
      <div class="importance-category" style="background: ${categoryColors[f.category]}22; color: ${categoryColors[f.category]}">
        ${f.category}
      </div>
    </div>
  `).join('');
}

// ── Results Rendering ────────────────────────────────────────────────────────
function renderResults(results) {
  lastResults = results;
  showSection('results-section');

  const { stage1, stage2 } = results;
  const disease = stage1.final_disease;

  // ── Patient Summary Card ──
  document.getElementById('result-patient-summary').innerHTML = `
    <div class="patient-badge">
      <span class="patient-icon">👤</span>
      <div>
        <div class="patient-name">${results.patient.gender === 'male' ? 'Male' : 'Female'} Patient, Age ${results.patient.age}</div>
        <div class="patient-meta">BMI: ${results.patient.bmi} · Glucose: ${results.patient.glucose} mg/dL · BP: ${results.patient.systolic}/${results.patient.diastolic} mmHg</div>
      </div>
    </div>
  `;

  // ── Stage 1: Classification ──
  const classContainer = document.getElementById('classification-result');
  classContainer.innerHTML = `
    <div class="disease-banner" style="border-color: ${disease.color}; box-shadow: 0 0 30px ${disease.color}22">
      <div class="disease-icon-large">${disease.icon}</div>
      <div class="disease-content">
        <div class="disease-title">${disease.label}</div>
        <div class="disease-desc">${disease.description}</div>
        <div class="disease-confidence">
          <div class="confidence-label">Ensemble Confidence</div>
          <div class="confidence-bar-wrap">
            <div class="confidence-bar" style="width: ${(stage1.ensemble.confidence * 100).toFixed(0)}%; background: ${disease.color}"></div>
          </div>
          <div class="confidence-value">${(stage1.ensemble.confidence * 100).toFixed(1)}%</div>
        </div>
      </div>
    </div>

    <div class="model-results-grid">
      ${renderModelCard(stage1.dnn, disease.color)}
      ${renderModelCard(stage1.random_forest, disease.color)}
      ${renderModelCard(stage1.gradient_boosting, disease.color)}
      ${renderModelCard(stage1.ensemble, disease.color, true)}
    </div>

    <div class="chart-container" style="height: 280px; margin-top: 1.5rem">
      <canvas id="classification-chart"></canvas>
    </div>
  `;

  // Render classification chart after DOM update
  requestAnimationFrame(() => {
    ChartManager.renderClassificationChart('classification-chart', results);
  });

  // ── Risk Factors ──
  if (disease.risk_factors.length > 0) {
    document.getElementById('risk-factors').innerHTML = `
      <h4 class="sub-heading">⚠️ Key Risk Factors Identified</h4>
      <div class="risk-tags">
        ${disease.risk_factors.map(r => `<span class="risk-tag">${r}</span>`).join('')}
      </div>
    `;
  }

  // ── Stage 2: Diet ──
  const dietDiv = document.getElementById('diet-recommendation');
  const diet = stage2.diet.recommended;
  dietDiv.innerHTML = `
    <div class="rec-card" style="border-color: ${diet.color}22">
      <div class="rec-header" style="background: linear-gradient(135deg, ${diet.color}22, transparent)">
        <span class="rec-icon">${diet.icon}</span>
        <div>
          <div class="rec-title">${diet.label}</div>
          <div class="rec-confidence">
            <div class="mini-bar" style="--pct: ${(stage2.diet.confidence * 100).toFixed(0)}%; --clr: ${diet.color}"></div>
            <span>${(stage2.diet.confidence * 100).toFixed(1)}% confidence</span>
          </div>
        </div>
      </div>
      <p class="rec-desc">${diet.description}</p>
      <div class="rec-details-grid">
        <div class="rec-detail-block">
          <div class="rec-detail-title">✅ Recommended Components</div>
          <ul>${diet.components.map(c => `<li>${c}</li>`).join('')}</ul>
        </div>
        <div class="rec-detail-block">
          <div class="rec-detail-title">🚫 Foods to Avoid</div>
          <ul>${diet.avoid.map(c => `<li>${c}</li>`).join('')}</ul>
        </div>
      </div>
      <div class="rec-footer">
        <span class="rec-stat">📊 ${diet.calories}</span>
        <span class="rec-stat">📚 ${diet.studies}</span>
      </div>
    </div>

    <div class="alt-recs">
      <div class="alt-title">Alternative Options</div>
      ${stage2.diet.alternatives.map(a => `
        <div class="alt-item">
          <span>${a.icon} ${a.label}</span>
          <div class="mini-bar-sm" style="--pct: ${(a.confidence * 100).toFixed(0)}%; --clr: ${a.color}"></div>
          <span class="alt-conf">${(a.confidence * 100).toFixed(1)}%</span>
        </div>
      `).join('')}
    </div>

    <div class="chart-container" style="height: 220px; margin-top: 1rem">
      <canvas id="diet-pie"></canvas>
    </div>
  `;

  // ── Stage 2: Drug ──
  const drugDiv = document.getElementById('drug-recommendation');
  const drug = stage2.drug.recommended;
  drugDiv.innerHTML = `
    <div class="rec-card" style="border-color: ${drug.color}22">
      <div class="rec-header" style="background: linear-gradient(135deg, ${drug.color}22, transparent)">
        <span class="rec-icon">${drug.icon}</span>
        <div>
          <div class="rec-title">${drug.label}</div>
          <div class="rec-confidence">
            <div class="mini-bar" style="--pct: ${(stage2.drug.confidence * 100).toFixed(0)}%; --clr: ${drug.color}"></div>
            <span>${(stage2.drug.confidence * 100).toFixed(1)}% confidence</span>
          </div>
        </div>
      </div>
      <p class="rec-desc">${drug.mechanism}</p>
      <div class="rec-details-grid">
        <div class="rec-detail-block">
          <div class="rec-detail-title">💊 Drug Examples</div>
          <ul>${drug.examples.map(e => `<li>${e}</li>`).join('')}</ul>
        </div>
        <div class="rec-detail-block">
          <div class="rec-detail-title">🚫 Contraindications</div>
          <ul>${(drug.contraindications.length > 0 ? drug.contraindications : ['None significant']).map(c => `<li>${c}</li>`).join('')}</ul>
        </div>
      </div>
      <div class="rec-footer">
        <span class="rec-stat">🎯 ${drug.indication}</span>
        <span class="rec-stat">🔬 ${drug.phase1_focus}</span>
      </div>
    </div>

    <div class="alt-recs">
      <div class="alt-title">Alternative Drug Classes</div>
      ${stage2.drug.alternatives.map(a => `
        <div class="alt-item">
          <span>${a.icon} ${a.label}</span>
          <div class="mini-bar-sm" style="--pct: ${(a.confidence * 100).toFixed(0)}%; --clr: ${a.color}"></div>
          <span class="alt-conf">${(a.confidence * 100).toFixed(1)}%</span>
        </div>
      `).join('')}
    </div>

    <div class="chart-container" style="height: 220px; margin-top: 1rem">
      <canvas id="drug-pie"></canvas>
    </div>
  `;

  // FIX: Always update radar chart after analysis (not just when EDA tab is open)
  ChartManager.renderRadar('radar-canvas', results.patient);

  // Render pie charts
  requestAnimationFrame(() => {
    ChartManager.renderDietPie('diet-pie', stage2.diet);
    ChartManager.renderDrugPie('drug-pie', stage2.drug);
  });

  // ── Medical Disclaimer ──
  const existingDisclaimer = document.getElementById('medical-disclaimer');
  if (!existingDisclaimer) {
    const disclaimer = document.createElement('div');
    disclaimer.id = 'medical-disclaimer';
    disclaimer.style.cssText = `
      margin-top: 2rem;
      padding: 1rem 1.25rem;
      background: rgba(245,158,11,0.08);
      border: 1px solid rgba(245,158,11,0.3);
      border-radius: 12px;
      font-size: 0.8rem;
      color: #f59e0b;
      line-height: 1.6;
    `;
    disclaimer.innerHTML = `
      <strong>⚠️ Research Use Only — Medical Disclaimer</strong><br>
      These recommendations are generated by a simulated ML model for <strong>Phase 1 clinical trial research purposes only</strong>.
      They do not constitute medical advice and should not replace consultation with a qualified healthcare professional.
      All drug and diet suggestions must be reviewed by a licensed physician before clinical application.
    `;
    document.querySelector('#results-section .main-content')?.appendChild(disclaimer);
  }
}

function renderModelCard(modelResult, color, isEnsemble = false) {
  const conf = (modelResult.confidence * 100).toFixed(1);
  return `
    <div class="model-card ${isEnsemble ? 'ensemble-card' : ''}">
      ${isEnsemble ? '<div class="ensemble-badge">Final</div>' : ''}
      <div class="model-name">${modelResult.model}</div>
      <div class="model-arch">${modelResult.architecture}</div>
      <div class="model-prediction">${modelResult.predicted_label}</div>
      <div class="model-conf-bar">
        <div class="model-conf-fill" style="width: ${conf}%; background: ${color}"></div>
      </div>
      <div class="model-conf-val">${conf}%</div>
    </div>
  `;
}

// ── Particles Background ─────────────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const particles = Array.from({ length: 55 }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 2 + 0.5,
    dx: (Math.random() - 0.5) * 0.4,
    dy: (Math.random() - 0.5) * 0.4,
    opacity: Math.random() * 0.5 + 0.1
  }));

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.x += p.dx;
      p.y += p.dy;
      if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.dy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(124, 58, 237, ${p.opacity})`;
      ctx.fill();
    });

    // Draw connections
    particles.forEach((p1, i) => {
      particles.slice(i + 1).forEach(p2 => {
        const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);
        if (dist < 100) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = `rgba(124, 58, 237, ${0.15 * (1 - dist / 100)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      });
    });

    requestAnimationFrame(animate);
  }
  animate();

  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
}

// ── Sample Data Loader ───────────────────────────────────────────────────────
function loadSamplePatient(type) {
  const samples = {
    t2dm: {
      age: 58, gender: 'female', ethnicity: 'south_asian',
      bmi: 29.5, glucose: 250, ldl: 158, hdl: 38, systolic: 148, diastolic: 92, heart_rate: 80,
      activity_level: 2, exercise_hours: 1, smoking: 'no', alcohol: 'none',
      family_history: true, history_diabetes: true, history_metabolic: false,
      history_hypertension: true, history_cvd: false, history_kidney: false,
      current_meds: 'insulin', allergies: 'none'
    },
    hypertension: {
      age: 62, gender: 'male', ethnicity: 'african_american',
      bmi: 27.5, glucose: 95, ldl: 140, hdl: 44, systolic: 168, diastolic: 102, heart_rate: 76,
      activity_level: 2, exercise_hours: 2, smoking: 'ex', alcohol: 'moderate',
      family_history: true, history_hypertension: true, history_diabetes: false,
      history_cvd: false, history_metabolic: false, history_kidney: false,
      current_meds: 'antihypertensive', allergies: 'none'
    },
    healthy: {
      age: 28, gender: 'male', ethnicity: 'caucasian',
      bmi: 22.1, glucose: 82, ldl: 90, hdl: 68, systolic: 112, diastolic: 72, heart_rate: 65,
      activity_level: 5, exercise_hours: 7, smoking: 'no', alcohol: 'none',
      family_history: false, history_diabetes: false, history_hypertension: false,
      history_cvd: false, history_metabolic: false, history_kidney: false,
      current_meds: 'none', allergies: 'none'
    },
    cvd: {
      age: 68, gender: 'male', ethnicity: 'caucasian',
      bmi: 27.8, glucose: 104, ldl: 198, hdl: 32, systolic: 145, diastolic: 90, heart_rate: 82,
      activity_level: 2, exercise_hours: 1, smoking: 'yes', alcohol: 'moderate',
      family_history: true, history_cvd: true, history_diabetes: false,
      history_hypertension: true, history_metabolic: false, history_kidney: false,
      current_meds: 'statins', allergies: 'none'
    },
    stroke: {
      age: 72, gender: 'male', ethnicity: 'caucasian',
      bmi: 31.5, glucose: 210, ldl: 180, hdl: 35, systolic: 175, diastolic: 105, heart_rate: 88,
      activity_level: 1, exercise_hours: 0, smoking: 'yes', alcohol: 'moderate',
      family_history: true, history_cvd: true, history_diabetes: false,
      history_hypertension: true, history_metabolic: false, history_kidney: false,
      current_meds: 'anticoagulant', allergies: 'none'
    }
  };

  const s = samples[type];
  if (!s) return;

  Object.entries(s).forEach(([key, val]) => {
    const el = document.getElementById(key);
    if (el) {
      if (el.type === 'checkbox') {
        el.checked = val;
        // FIX: dispatch change event so visual .checked class syncs on the label
        el.dispatchEvent(new Event('change', { bubbles: true }));
      } else {
        el.value = val;
      }

      // Trigger slider update
      if (el.type === 'range') {
        el.dispatchEvent(new Event('input'));
      }
    }
  });

  showToast(`Loaded ${type.toUpperCase()} sample patient data`, 'success');
  goToStep(1);
}

// ── Main Submit Handler ───────────────────────────────────────────────────────
async function handleSubmit() {
  if (!validateStep(currentStep)) return;

  patientData = collectFormData();
  showLoading();

  try {
    const results = await mlEngine.predict(patientData);
    hideLoading();
    renderResults(results);
    showToast('Analysis complete! Recommendations generated.', 'success');
  } catch (err) {
    hideLoading();
    console.error(err);
    // Show validation errors to the user in the toast
    const msg = err.message.startsWith('Validation failed')
      ? err.message.replace('Validation failed:\n', '')
      : 'Analysis failed. Please check your inputs and try again.';
    showToast(msg, 'error');
  }
}

// ── PDF / Print Export ───────────────────────────────────────────────────────
function exportResults() {
  if (!lastResults) {
    showToast('Run an analysis first to export results.', 'error');
    return;
  }
  const { stage1, stage2, patient } = lastResults;
  const disease = stage1.final_disease;
  const diet = stage2.diet.recommended;
  const drug = stage2.drug.recommended;

  const html = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>ClinicalML — Patient Report</title>
      <style>
        body { font-family: Arial, sans-serif; padding: 40px; color: #1e293b; }
        h1 { color: #7c3aed; border-bottom: 2px solid #7c3aed; padding-bottom: 8px; }
        h2 { color: #475569; margin-top: 28px; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 100px;
                 background: #f1f5f9; font-size: 0.85em; margin: 2px; }
        .section { margin-bottom: 24px; padding: 16px; background: #f8fafc;
                   border-left: 4px solid #7c3aed; border-radius: 4px; }
        .disclaimer { margin-top: 32px; padding: 12px; background: #fffbeb;
                      border: 1px solid #f59e0b; border-radius: 6px;
                      font-size: 0.8em; color: #92400e; }
        table { border-collapse: collapse; width: 100%; }
        td, th { padding: 8px 12px; border: 1px solid #e2e8f0; text-align: left; }
        th { background: #f1f5f9; }
      </style>
    </head>
    <body>
      <h1>ClinicalML — Patient Analysis Report</h1>
      <p>Generated: ${new Date().toLocaleString()}</p>

      <h2>Patient Summary</h2>
      <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Age</td><td>${patient.age} years</td></tr>
        <tr><td>Gender</td><td>${patient.gender}</td></tr>
        <tr><td>BMI</td><td>${patient.bmi} kg/m²</td></tr>
        <tr><td>Fasting Glucose</td><td>${patient.glucose} mg/dL</td></tr>
        <tr><td>LDL</td><td>${patient.ldl} mg/dL</td></tr>
        <tr><td>HDL</td><td>${patient.hdl} mg/dL</td></tr>
        <tr><td>Blood Pressure</td><td>${patient.systolic}/${patient.diastolic} mmHg</td></tr>
      </table>

      <h2>Stage 1 — Disease Classification</h2>
      <div class="section">
        <strong>${disease.icon} ${disease.label}</strong><br>
        ${disease.description}<br><br>
        Ensemble Confidence: <strong>${(stage1.ensemble.confidence * 100).toFixed(1)}%</strong>
        (DNN: ${(stage1.dnn.confidence * 100).toFixed(1)}%,
        RF: ${(stage1.random_forest.confidence * 100).toFixed(1)}%,
        GB: ${(stage1.gradient_boosting.confidence * 100).toFixed(1)}%)
      </div>

      <h2>Stage 2 — Diet Recommendation</h2>
      <div class="section">
        <strong>${diet.icon} ${diet.label}</strong> — Confidence: ${(stage2.diet.confidence * 100).toFixed(1)}%<br>
        ${diet.description}<br><br>
        <strong>Recommended components:</strong> ${diet.components.join(', ')}<br>
        <strong>Avoid:</strong> ${diet.avoid.join(', ')}
      </div>

      <h2>Stage 2 — Drug Class Recommendation</h2>
      <div class="section">
        <strong>${drug.icon} ${drug.label}</strong> — Confidence: ${(stage2.drug.confidence * 100).toFixed(1)}%<br>
        ${drug.mechanism}<br><br>
        <strong>Examples:</strong> ${drug.examples.join(', ')}<br>
        <strong>Contraindications:</strong> ${drug.contraindications.join(', ') || 'None significant'}
      </div>

      <div class="disclaimer">
        <strong>⚠️ Research Use Only:</strong> This report is generated by a simulated ML model
        for Phase 1 clinical trial research purposes only. It does not constitute medical advice.
        All recommendations must be reviewed by a licensed physician before clinical application.
      </div>
    </body>
    </html>
  `;

  const win = window.open('', '_blank');
  win.document.write(html);
  win.document.close();
  setTimeout(() => win.print(), 500);
}

// ── Initialization ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initSliders();
  updateNavButtons();
  showSection('input-section');

  // Nav tabs
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => showSection(tab.dataset.section));
  });

  // Step buttons
  document.getElementById('prev-btn')?.addEventListener('click', () => goToStep(currentStep - 1));
  document.getElementById('next-btn')?.addEventListener('click', () => goToStep(currentStep + 1));
  document.getElementById('submit-btn')?.addEventListener('click', handleSubmit);

  // Hero CTA
  document.getElementById('cta-btn')?.addEventListener('click', () => {
    showSection('input-section');
  });

  // Sample buttons
  document.querySelectorAll('[data-sample]').forEach(btn => {
    btn.addEventListener('click', () => loadSamplePatient(btn.dataset.sample));
  });

  // Number input range enforcement
  document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('blur', () => {
      const min = parseFloat(input.min);
      const max = parseFloat(input.max);
      if (!isNaN(min) && parseFloat(input.value) < min) input.value = min;
      if (!isNaN(max) && parseFloat(input.value) > max) input.value = max;
    });
  });

  // Initial EDA charts on hero section
  setTimeout(() => {
    ChartManager.renderFeatureImportance('hero-importance-canvas');
  }, 300);
});
