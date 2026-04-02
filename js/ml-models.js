// ═══════════════════════════════════════════════════════════════════════════════
// ClinicalML — Real Model Inference Engine (Pure JavaScript)
// ═══════════════════════════════════════════════════════════════════════════════
// Loads trained model JSON files (RF, GB, MLP) exported by train_models.py
// and runs inference entirely in the browser. No backend or ONNX needed.
// ═══════════════════════════════════════════════════════════════════════════════

class RealModelEngine {
  constructor() {
    this.rf = null;
    this.gb = null;
    this.dnn = null;
    this.metadata = null;
    this.loaded = false;
  }

  // ── Load all model files ──────────────────────────────────────────────────
  async loadModels() {
    if (this.loaded) return;
    console.time('Model loading');

    // Cache-busting to avoid stale HTML responses from previous SPA config
    const cacheBust = `?v=${Date.now()}`;
    const fetchJSON = async (url) => {
      const resp = await fetch(url + cacheBust, { cache: 'no-cache' });
      if (!resp.ok) throw new Error(`Failed to fetch ${url}: HTTP ${resp.status}`);
      const contentType = resp.headers.get('content-type') || '';
      if (!contentType.includes('json') && !contentType.includes('octet')) {
        const text = await resp.text();
        if (text.trimStart().startsWith('<')) {
          throw new Error(`${url} returned HTML instead of JSON. Clear browser cache and reload.`);
        }
        return JSON.parse(text);
      }
      return resp.json();
    };

    const [rfData, gbData, dnnData, meta] = await Promise.all([
      fetchJSON('models/rf_model.json'),
      fetchJSON('models/gb_model.json'),
      fetchJSON('models/dnn_weights.json'),
      fetchJSON('models/metadata.json')
    ]);

    this.rf = rfData;
    this.gb = gbData;
    this.dnn = dnnData;
    this.metadata = meta;
    this.loaded = true;

    console.timeEnd('Model loading');
    console.log(`Loaded: RF (${rfData.trees.length} trees), GB (${gbData.estimators.length} rounds), DNN (${dnnData.layers.length} layers)`);
  }

  // ── Feature extraction from form data ─────────────────────────────────────
  extractFeatures(patient) {
    // Map form inputs to the 7 features the model was trained on:
    // [age, gender_enc, bmi, blood_glucose, hypertension, heart_disease_history, smoking_enc]
    const genderEnc = (patient.gender === 'male') ? 1 : 0;
    const hypFlag = (patient.systolic > 140 || patient.diastolic > 90 || patient.history_hypertension) ? 1 : 0;
    const cvdFlag = patient.history_cvd ? 1 : 0;
    const smokingEnc = patient.smoking === 'yes' ? 2 : patient.smoking === 'ex' ? 1 : 0;

    return [
      patient.age,
      genderEnc,
      patient.bmi,
      patient.glucose,
      hypFlag,
      cvdFlag,
      smokingEnc
    ];
  }

  // ── Standardize features using trained scaler params ──────────────────────
  standardize(features) {
    const { mean, scale } = this.metadata.scaler;
    return features.map((v, i) => (v - mean[i]) / scale[i]);
  }

  // ── Random Forest inference ───────────────────────────────────────────────
  predictRF(scaledFeatures) {
    const { trees, n_classes } = this.rf;
    const votes = new Float64Array(n_classes);

    for (const tree of trees) {
      let nodeIdx = 0;
      // Traverse tree until leaf
      while (!tree[nodeIdx].leaf) {
        const node = tree[nodeIdx];
        nodeIdx = scaledFeatures[node.f] <= node.t ? node.l : node.r;
      }
      const probs = tree[nodeIdx].leaf;
      for (let c = 0; c < n_classes; c++) votes[c] += probs[c];
    }

    // Average across trees
    const total = votes.reduce((a, b) => a + b, 0);
    return Array.from(votes).map(v => v / total);
  }

  // ── Gradient Boosting inference ───────────────────────────────────────────
  predictGB(scaledFeatures) {
    const { estimators, n_classes, learning_rate, init: initScores } = this.gb;
    const scores = new Float64Array(n_classes);

    // Initialize with log-prior
    for (let c = 0; c < n_classes; c++) scores[c] = initScores[c];

    // Accumulate residual predictions
    for (const round of estimators) {
      for (let c = 0; c < n_classes; c++) {
        const tree = round[c];
        let nodeIdx = 0;
        while (tree[nodeIdx].leaf === undefined) {
          const node = tree[nodeIdx];
          nodeIdx = scaledFeatures[node.f] <= node.t ? node.l : node.r;
        }
        scores[c] += learning_rate * tree[nodeIdx].leaf;
      }
    }

    // Softmax to get probabilities
    return this._softmax(Array.from(scores));
  }

  // ── MLP (DNN) inference ───────────────────────────────────────────────────
  predictDNN(scaledFeatures) {
    let x = scaledFeatures.slice(); // copy

    for (const layer of this.dnn.layers) {
      const { weights, biases, activation } = layer;
      const inputSize = weights.length;
      const outputSize = weights[0].length;
      const out = new Array(outputSize).fill(0);

      // Matrix multiply: out = x * W + b
      for (let j = 0; j < outputSize; j++) {
        let sum = biases[j];
        for (let i = 0; i < inputSize; i++) {
          sum += x[i] * weights[i][j];
        }
        out[j] = sum;
      }

      // Activation
      if (activation === 'relu') {
        for (let j = 0; j < outputSize; j++) {
          out[j] = Math.max(0, out[j]);
        }
      } else if (activation === 'softmax') {
        const sm = this._softmax(out);
        for (let j = 0; j < outputSize; j++) out[j] = sm[j];
      }

      x = out;
    }

    return x;
  }

  // ── Ensemble: weighted average of all 3 models ────────────────────────────
  predictEnsemble(patient) {
    const raw = this.extractFeatures(patient);
    const scaled = this.standardize(raw);

    const rfProbs  = this.predictRF(scaled);
    const gbProbs  = this.predictGB(scaled);
    const dnnProbs = this.predictDNN(scaled);

    // Weighted average: GB 40%, RF 35%, DNN 25% (GB had best accuracy)
    const n = rfProbs.length;
    const ensemble = new Array(n);
    for (let i = 0; i < n; i++) {
      ensemble[i] = 0.35 * rfProbs[i] + 0.40 * gbProbs[i] + 0.25 * dnnProbs[i];
    }

    // Normalize
    const total = ensemble.reduce((a, b) => a + b, 0);
    for (let i = 0; i < n; i++) ensemble[i] /= total;

    const classNames = this.metadata.class_names;
    const predictedIdx = ensemble.indexOf(Math.max(...ensemble));

    return {
      classNames,
      probabilities: ensemble,
      rfProbs,
      gbProbs,
      dnnProbs,
      predictedClass: classNames[predictedIdx],
      confidence: ensemble[predictedIdx],
      predictedIdx
    };
  }

  // ── Softmax utility ───────────────────────────────────────────────────────
  _softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }
}

// Global instance
const realModels = new RealModelEngine();
