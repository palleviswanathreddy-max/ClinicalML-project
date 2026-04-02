// ─── ML Engine: Two-Stage Pipeline Simulation ────────────────────────────────
// Stage 1: Disease Classification (DNN + Ensemble)
// Stage 2: Dual Recommendation Engine (Diet + Drug)

class MLEngine {
  constructor() {
    this.normalRanges = {
      bmi: { low: 18.5, high: 24.9 },
      glucose: { low: 70, high: 99 },
      ldl: { low: 0, high: 100 },
      hdl: { low: 40, high: 999 },
      systolic: { low: 90, high: 120 },
      diastolic: { low: 60, high: 80 },
      heart_rate: { low: 60, high: 100 }
    };
  }

  // ── Utility: Sigmoid & Softmax ──────────────────────────────────────────
  sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  softmax(scores) {
    const max = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }

  // Normalize a value to [0, 1] given min/max
  normalize(val, min, max) {
    return Math.max(0, Math.min(1, (val - min) / (max - min)));
  }

  // ── Feature Engineering ─────────────────────────────────────────────────
  // ── Input Validation ────────────────────────────────────────────────────
  validate(patient) {
    const errors = [];
    if (!patient.bmi || patient.bmi < 10 || patient.bmi > 80)
      errors.push('BMI must be between 10 and 80 kg/m².');
    if (!patient.glucose || patient.glucose < 30 || patient.glucose > 600)
      errors.push('Fasting glucose must be between 30 and 600 mg/dL.');
    if (!patient.ldl || patient.ldl < 10 || patient.ldl > 500)
      errors.push('LDL must be between 10 and 500 mg/dL.');
    if (!patient.hdl || patient.hdl < 10 || patient.hdl > 200)
      errors.push('HDL must be between 10 and 200 mg/dL.');
    if (!patient.systolic || patient.systolic < 60 || patient.systolic > 260)
      errors.push('Systolic BP must be between 60 and 260 mmHg.');
    if (!patient.diastolic || patient.diastolic < 30 || patient.diastolic > 160)
      errors.push('Diastolic BP must be between 30 and 160 mmHg.');
    if (!patient.age || patient.age < 1 || patient.age > 120)
      errors.push('Age must be between 1 and 120 years.');
    return errors;
  }

  extractFeatures(patient) {
    const age = patient.age || 45;
    const bmi = patient.bmi || 25;
    const glucose = patient.glucose || 90;
    const ldl = patient.ldl || 120;
    const hdl = patient.hdl || 55;
    const systolic = patient.systolic || 120;
    const diastolic = patient.diastolic || 80;
    const heartRate = patient.heart_rate || 72;
    const activityLevel = patient.activity_level || 3;   // 1–5
    const exerciseHours = patient.exercise_hours || 3;
    const smoking = patient.smoking === 'yes' ? 1 : patient.smoking === 'ex' ? 0.5 : 0;
    const alcohol = patient.alcohol === 'heavy' ? 1 : patient.alcohol === 'moderate' ? 0.5 : 0;
    const familyHistory = patient.family_history ? 1 : 0;
    const historyDiabetes = patient.history_diabetes ? 1 : 0;
    const historyHypertension = patient.history_hypertension ? 1 : 0;
    const historyCvd = patient.history_cvd ? 1 : 0;
    const historyMetabolic = patient.history_metabolic ? 1 : 0;
    const historyKidney = patient.history_kidney ? 1 : 0;
    const currentMeds = patient.current_meds || 'none';
    const gender = patient.gender === 'male' ? 1 : 0;

    // Severity scores (0 = normal, 1 = severe)
    const bmiScore = bmi < 18.5 ? 0.2 :
                     bmi < 25 ? 0 :
                     bmi < 30 ? this.normalize(bmi, 25, 40) :
                     this.normalize(bmi, 25, 50);

    const glucoseScore = glucose < 100 ? 0 :
                         glucose < 126 ? this.normalize(glucose, 100, 200) :
                         this.normalize(glucose, 100, 300);

    const ldlScore = ldl < 100 ? 0 :
                     ldl < 130 ? this.normalize(ldl, 100, 200) :
                     this.normalize(ldl, 80, 250);

    // FIX: HDL args were reversed — normalize(hdl, low, high) not normalize(low, hdl, high)
    const hdlScore = hdl >= 60 ? 0 :
                     hdl >= 40 ? this.normalize(hdl, 40, 60) :
                     1 - this.normalize(hdl, 20, 60);

    const bpScore = this.normalize(
      Math.max(systolic - 120, 0) * 0.6 + Math.max(diastolic - 80, 0) * 0.4,
      0, 80
    );

    const activityScore = 1 - this.normalize(activityLevel, 1, 5);
    const exerciseScore = 1 - this.normalize(exerciseHours, 0, 10);
    const ageScore = this.normalize(age, 20, 80);

    // Medical history bonus scores (boost disease-specific signal)
    const medHistoryScore = historyDiabetes * 0.3 + historyHypertension * 0.25 +
                            historyCvd * 0.25 + historyMetabolic * 0.2 + historyKidney * 0.15;

    return {
      age, bmi, glucose, ldl, hdl, systolic, diastolic, heartRate,
      activityLevel, exerciseHours, smoking, alcohol, familyHistory, gender,
      historyDiabetes, historyHypertension, historyCvd, historyMetabolic, historyKidney,
      currentMeds,
      bmiScore, glucoseScore, ldlScore, hdlScore, bpScore,
      activityScore, exerciseScore, ageScore, medHistoryScore
    };
  }

  // ── Stage 1a: DNN Simulation ────────────────────────────────────────────
  runDNN(features) {
    const { bmiScore, glucoseScore, ldlScore, hdlScore, bpScore,
            activityScore, ageScore, smoking, alcohol, familyHistory, gender,
            historyDiabetes, historyHypertension, historyCvd, medHistoryScore } = features;

    // Layer 1: deterministic dense activations (dropout replaced with fixed 0.9 scale = training inference mode)
    const h1 = [
      this.sigmoid(glucoseScore * 2.8 + bmiScore * 1.9 + activityScore * 1.2 + familyHistory * 0.9 + historyDiabetes * 1.2 - 0.5),
      this.sigmoid(bpScore * 3.1 + ageScore * 1.5 + smoking * 1.1 + alcohol * 0.7 + historyHypertension * 1.1 - 0.4),
      this.sigmoid(ldlScore * 2.7 + hdlScore * 2.1 + smoking * 1.4 + ageScore * 0.8 + historyCvd * 1.2 - 0.3),
      this.sigmoid(bmiScore * 2.5 + glucoseScore * 1.8 + ldlScore * 1.1 + hdlScore * 0.9 - 0.6),
      this.sigmoid(activityScore * 2.2 + bmiScore * 2.0 + ageScore * 0.7 + alcohol * 0.6 - 0.4),
      this.sigmoid(ageScore * 0.3 + activityScore * 0.3 + bmiScore * 0.2 + glucoseScore * 0.2 - 2.5)
    ];

    // Inference mode: use fixed dropout scale (0.9) instead of random
    const h1d = h1.map(v => v * 0.9);

    // Layer 2 (output logits for 6 classes)
    const logits = [
      h1d[0] * 3.2 + h1d[3] * 2.4 + h1d[4] * 1.1,   // T2DM
      h1d[1] * 3.5 + h1d[0] * 0.8 + h1d[2] * 0.6,   // Hypertension
      h1d[2] * 3.3 + h1d[1] * 1.9 + h1d[0] * 0.7,   // CVD
      h1d[3] * 3.0 + h1d[0] * 1.6 + h1d[4] * 1.2,   // Metabolic Syndrome
      h1d[4] * 3.1 + h1d[3] * 1.8 + h1d[0] * 0.9,   // Obesity-related
      h1d[5] * 4.0 - h1d[0] * 1.5 - h1d[1] * 1.0 - h1d[2] * 1.0  // Healthy
    ];

    const probabilities = this.softmax(logits);
    const maxIdx = probabilities.indexOf(Math.max(...probabilities));

    return {
      model: "Deep Neural Network",
      architecture: "4-layer DNN (BN + L2 + Dropout)",
      probabilities,
      predicted_class: DISEASE_CLASSES[maxIdx].id,
      predicted_label: DISEASE_CLASSES[maxIdx].label,
      confidence: probabilities[maxIdx]
    };
  }

  // ── Stage 1b: Random Forest Simulation ─────────────────────────────────
  // Deterministic pseudo-random via seeded LCG so results are stable per-patient
  _lcg(seed) {
    let s = seed;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  }

  runRandomForest(features) {
    const { bmiScore, glucoseScore, ldlScore, hdlScore, bpScore,
            activityScore, ageScore, smoking, familyHistory,
            historyDiabetes, historyHypertension, historyCvd, historyMetabolic } = features;

    // Deterministic seed derived from patient values (reproducible per patient)
    const seed = Math.round(
      (glucoseScore + bmiScore + ldlScore + hdlScore + bpScore + ageScore) * 1e6
    ) % 0xffffffff;
    const rand = this._lcg(seed);

    const n_trees = 100;
    const votes = new Array(6).fill(0);

    for (let t = 0; t < n_trees; t++) {
      // Deterministic per-tree noise via seeded random
      const gScore = glucoseScore + (rand() - 0.5) * 0.15;
      const bScore = bmiScore     + (rand() - 0.5) * 0.15;
      const lScore = ldlScore     + (rand() - 0.5) * 0.12;
      const hScore = hdlScore     + (rand() - 0.5) * 0.12;
      const pScore = bpScore      + (rand() - 0.5) * 0.15;
      const aScore = activityScore + (rand() - 0.5) * 0.10;

      // Expanded decision tree (reduces over-classification as Healthy)
      if (gScore > 0.45 && bScore > 0.30) {
        votes[0]++;                                    // T2DM
      } else if (gScore > 0.35 && historyDiabetes) {
        votes[0]++;                                    // T2DM (history-boosted)
      } else if (pScore > 0.35 && bScore > 0.20) {
        votes[1]++;                                    // Hypertension
      } else if (pScore > 0.30 && historyHypertension) {
        votes[1]++;                                    // Hypertension (history-boosted)
      } else if (lScore > 0.40 && hScore > 0.35) {
        votes[2]++;                                    // CVD
      } else if (lScore > 0.30 && historyCvd) {
        votes[2]++;                                    // CVD (history-boosted)
      } else if (gScore > 0.25 && lScore > 0.25 && bScore > 0.35) {
        votes[3]++;                                    // Metabolic Syndrome
      } else if (historyMetabolic && bScore > 0.20) {
        votes[3]++;                                    // Metabolic (history-boosted)
      } else if (bScore > 0.50 && aScore > 0.45) {
        votes[4]++;                                    // Obesity-related
      } else if (bScore > 0.40 && gScore > 0.20 && aScore > 0.55) {
        votes[4]++;                                    // Obesity (secondary branch)
      } else {
        votes[5]++;                                    // Healthy
      }
    }

    const total = votes.reduce((a, b) => a + b, 0);
    const probabilities = votes.map(v => v / total);
    const maxIdx = probabilities.indexOf(Math.max(...probabilities));

    return {
      model: "Random Forest",
      architecture: "100 trees, max_depth=8, min_samples_split=5",
      probabilities,
      predicted_class: DISEASE_CLASSES[maxIdx].id,
      predicted_label: DISEASE_CLASSES[maxIdx].label,
      confidence: probabilities[maxIdx]
    };
  }

  // ── Stage 1c: Gradient Boosting Simulation ──────────────────────────────
  runGradientBoosting(features) {
    const { bmiScore, glucoseScore, ldlScore, hdlScore, bpScore,
            activityScore, ageScore, smoking, familyHistory,
            historyDiabetes, historyHypertension, historyCvd, medHistoryScore } = features;

    // GB tends to have higher confidence but similar direction to RF
    const baseScores = [
      glucoseScore * 0.40 + bmiScore * 0.30 + activityScore * 0.15 + familyHistory * 0.10,  // T2DM
      bpScore * 0.45 + ageScore * 0.25 + smoking * 0.20 + bmiScore * 0.10,                   // HTN
      ldlScore * 0.40 + hdlScore * 0.30 + smoking * 0.20 + ageScore * 0.10,                  // CVD
      glucoseScore * 0.25 + bmiScore * 0.30 + ldlScore * 0.25 + hdlScore * 0.20,             // Metabolic
      bmiScore * 0.45 + activityScore * 0.35 + ageScore * 0.10 + glucoseScore * 0.10,        // Obesity
      Math.max(0, 0.8 - glucoseScore * 0.5 - bmiScore * 0.4 - bpScore * 0.3)                // Healthy
    ];

    // Add gradient boosting correction (deterministic boost factor)
    const boosted = baseScores.map(s => s * 1.18);
    const probabilities = this.softmax(boosted);
    const maxIdx = probabilities.indexOf(Math.max(...probabilities));

    return {
      model: "Gradient Boosting",
      architecture: "n_estimators=200, lr=0.05, max_depth=5",
      probabilities,
      predicted_class: DISEASE_CLASSES[maxIdx].id,
      predicted_label: DISEASE_CLASSES[maxIdx].label,
      confidence: probabilities[maxIdx]
    };
  }

  // ── Stage 1d: Ensemble (Soft Voting) — DNN + RF + GB ────────────────────
  runEnsemble(dnnResult, rfResult, gbResult) {
    // FIX: Include DNN in ensemble (was excluded before)
    // Weights: DNN=35%, RF=35%, GB=30%
    const probabilities = dnnResult.probabilities.map(
      (p, i) => p * 0.35 + rfResult.probabilities[i] * 0.35 + gbResult.probabilities[i] * 0.30
    );
    // Re-normalize
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const normalized = probabilities.map(p => p / sum);
    const maxIdx = normalized.indexOf(Math.max(...normalized));

    return {
      model: "Ensemble (Soft Voting)",
      architecture: "DNN(35%) + RF(35%) + GB(30%) soft voting",
      probabilities: normalized,
      predicted_class: DISEASE_CLASSES[maxIdx].id,
      predicted_label: DISEASE_CLASSES[maxIdx].label,
      confidence: normalized[maxIdx]
    };
  }

  // ── Stage 2: Diet Recommendation Engine ─────────────────────────────────
  recommendDiet(diseaseClass, features) {
    const { bmiScore, glucoseScore, ldlScore, hdlScore, bpScore } = features;

    const scores = {
      mediterranean: 0,
      dash: 0,
      low_glycemic: 0,
      heart_healthy: 0,
      ketogenic: 0,
      balanced: 0
    };

    switch (diseaseClass) {
      case 'hypertension':
        scores.dash += 0.55;
        scores.mediterranean += 0.25;
        scores.heart_healthy += 0.15;
        break;
      case 'cvd':
        scores.heart_healthy += 0.50;
        scores.mediterranean += 0.28;
        scores.dash += 0.12;
        break;
      case 't2dm':
        scores.low_glycemic += 0.50;
        scores.mediterranean += 0.25;
        scores.ketogenic += 0.15;
        break;
      case 'metabolic':
        scores.mediterranean += 0.40;
        scores.low_glycemic += 0.30;
        scores.dash += 0.20;
        break;
      case 'obesity':
        scores.ketogenic += 0.40;
        scores.low_glycemic += 0.30;
        scores.mediterranean += 0.20;
        break;
      case 'stroke':
        scores.dash += 0.45;
        scores.mediterranean += 0.30;
        scores.heart_healthy += 0.15;
        break;
      case 'healthy':
        scores.balanced += 0.55;
        scores.mediterranean += 0.30;
        break;
    }

    // Feature-based adjustments
    if (bpScore > 0.4) { scores.dash += 0.1; scores.mediterranean += 0.05; }
    if (ldlScore > 0.4) { scores.heart_healthy += 0.1; scores.mediterranean += 0.05; }
    if (glucoseScore > 0.4) { scores.low_glycemic += 0.1; scores.ketogenic += 0.05; }
    if (bmiScore > 0.5) { scores.ketogenic += 0.08; scores.low_glycemic += 0.06; }

    // Softmax normalization
    const vals = Object.values(scores);
    const probs = this.softmax(vals.map(v => v * 5));
    const keys = Object.keys(scores);

    const result = keys.map((k, i) => ({ key: k, confidence: probs[i] }))
                       .sort((a, b) => b.confidence - a.confidence);

    const top = result[0];
    return {
      recommended: DIET_PLANS[top.key],
      confidence: top.confidence,
      alternatives: result.slice(1, 3).map(r => ({
        ...DIET_PLANS[r.key],
        confidence: r.confidence
      })),
      all_scores: result.map(r => ({ label: DIET_PLANS[r.key].label, confidence: r.confidence }))
    };
  }

  // ── Stage 2: Drug Recommendation Engine ─────────────────────────────────
  recommendDrug(diseaseClass, features) {
    const { bmiScore, glucoseScore, ldlScore, hdlScore, bpScore, ageScore } = features;

    const scores = {
      ace_inhibitors: 0,
      statins: 0,
      metformin_class: 0,
      beta_blockers: 0,
      glp1_agonists: 0,
      no_intervention: 0
    };

    switch (diseaseClass) {
      case 'hypertension':
        scores.ace_inhibitors += 0.52;
        scores.beta_blockers += 0.28;
        scores.no_intervention += 0.10;
        break;
      case 'cvd':
        scores.statins += 0.50;
        scores.beta_blockers += 0.25;
        scores.ace_inhibitors += 0.15;
        break;
      case 't2dm':
        scores.metformin_class += 0.55;
        scores.glp1_agonists += 0.25;
        scores.no_intervention += 0.10;
        break;
      case 'metabolic':
        scores.metformin_class += 0.35;
        scores.statins += 0.25;
        scores.glp1_agonists += 0.22;
        break;
      case 'obesity':
        scores.glp1_agonists += 0.48;
        scores.metformin_class += 0.28;
        scores.no_intervention += 0.14;
        break;
      case 'stroke':
        scores.statins += 0.40;
        scores.ace_inhibitors += 0.30;
        scores.beta_blockers += 0.20;
        break;
      case 'healthy':
        scores.no_intervention += 0.75;
        break;
    }

    // Feature adjustments
    if (ldlScore > 0.45) { scores.statins += 0.12; }
    if (bpScore > 0.45) { scores.ace_inhibitors += 0.10; scores.beta_blockers += 0.08; }
    if (glucoseScore > 0.45) { scores.metformin_class += 0.10; scores.glp1_agonists += 0.08; }
    if (bmiScore > 0.55 && glucoseScore > 0.3) { scores.glp1_agonists += 0.09; }

    const vals = Object.values(scores);
    const probs = this.softmax(vals.map(v => v * 5.5));
    const keys = Object.keys(scores);

    const result = keys.map((k, i) => ({ key: k, confidence: probs[i] }))
                       .sort((a, b) => b.confidence - a.confidence);

    const top = result[0];
    return {
      recommended: DRUG_CLASSES[top.key],
      confidence: top.confidence,
      alternatives: result.slice(1, 3).map(r => ({
        ...DRUG_CLASSES[r.key],
        confidence: r.confidence
      })),
      all_scores: result.map(r => ({ label: DRUG_CLASSES[r.key].label, confidence: r.confidence }))
    };
  }

  // ── Full Pipeline (Real Trained Models) ──────────────────────────────────
  async predict(patientData) {
    // Input validation
    const validationErrors = this.validate(patientData);
    if (validationErrors.length > 0) {
      throw new Error('Validation failed:\n' + validationErrors.join('\n'));
    }

    // Load real trained models on first call
    try {
      await realModels.loadModels();
    } catch (loadErr) {
      console.error('Model loading failed:', loadErr);
      throw new Error('Failed to load ML models. Please hard-refresh (Ctrl+Shift+R) and try again. Error: ' + loadErr.message);
    }

    // Brief processing delay for UX
    await new Promise(r => setTimeout(r, 1200));

    // ── Stage 1: Real Model Inference ──
    const raw = realModels.extractFeatures(patientData);
    const scaled = realModels.standardize(raw);

    const rfProbs  = realModels.predictRF(scaled);
    const gbProbs  = realModels.predictGB(scaled);
    const dnnProbs = realModels.predictDNN(scaled);
    const ensRes   = realModels.predictEnsemble(patientData);

    const classNames = realModels.metadata.class_names;
    const metrics = realModels.metadata.metrics;

    // Build per-model result objects matching UI expectations
    const makeResult = (probs, modelName, arch) => {
      const maxIdx = probs.indexOf(Math.max(...probs));
      const diseaseId = classNames[maxIdx];
      const diseaseObj = DISEASE_CLASSES.find(d => d.id === diseaseId) || DISEASE_CLASSES[DISEASE_CLASSES.length - 1];
      return {
        model: modelName,
        architecture: arch,
        probabilities: probs,
        predicted_class: diseaseId,
        predicted_label: diseaseObj.label,
        confidence: probs[maxIdx]
      };
    };

    const dnnResult = makeResult(dnnProbs, "Deep Neural Network",
      `MLP (64\u219232\u219216), Acc: ${(metrics.mlp.accuracy * 100).toFixed(1)}%`);
    const rfResult  = makeResult(rfProbs, "Random Forest",
      `30 trees, depth=8, Acc: ${(metrics.rf.accuracy * 100).toFixed(1)}%`);
    const gbResult  = makeResult(gbProbs, "Gradient Boosting",
      `80 est, depth=4, Acc: ${(metrics.gb.accuracy * 100).toFixed(1)}%`);
    const ensembleResult = makeResult(ensRes.probabilities, "Ensemble (Soft Voting)",
      `DNN(25%) + RF(35%) + GB(40%) weighted`);

    const finalClass = ensembleResult.predicted_class;
    const finalDisease = DISEASE_CLASSES.find(d => d.id === finalClass) || DISEASE_CLASSES[DISEASE_CLASSES.length - 1];

    // ── Stage 2: Rule-based recommendations ──
    const features = this.extractFeatures(patientData);
    const dietRec = this.recommendDiet(finalClass, features);
    const drugRec = this.recommendDrug(finalClass, features);

    return {
      patient: patientData,
      features,
      stage1: {
        dnn: dnnResult,
        random_forest: rfResult,
        gradient_boosting: gbResult,
        ensemble: ensembleResult,
        final_disease: finalDisease,
        final_class: finalClass
      },
      stage2: {
        diet: dietRec,
        drug: drugRec
      }
    };
  }
}

const mlEngine = new MLEngine();

