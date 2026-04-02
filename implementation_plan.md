# Real Dataset Integration — ClinicalML

## Problem
Currently the ML pipeline uses **hardcoded rules** (`ml-engine.js`) to simulate disease classification and drug/diet recommendations. There are no trained models — results are faked. We want to use the **real datasets already in the project** to train actual models and deliver real predictions.

## Your Datasets

| Dataset | File | Rows | Features | Target |
|---------|------|------|----------|--------|
| **Diabetes** | `diabetes_prediction_dataset.csv` | 100,000 | age, gender, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level | `diabetes` (0/1) |
| **Heart Disease** | `heart_cleveland_upload.csv` | 297 | age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal | `condition` (0/1) |
| **Stroke** | `healthcare-dataset-stroke-data.csv` | 5,110 | gender, age, hypertension, heart_disease, ever_married, work_type, bmi, avg_glucose_level, smoking_status | `stroke` (0/1) |
| **Lung Cancer** | `survey lung cancer.csv` | 309 | age, gender, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, etc. | `LUNG_CANCER` (YES/NO) |

---

## User Review Required

> [!IMPORTANT]
> **Architecture Choice: Browser-only (no Python server needed at runtime)**
>
> I propose training models in Python **once**, exporting them as ONNX files, and loading them in the browser with `onnxruntime-web`. This means:
> - ✅ No backend server to deploy/maintain
> - ✅ Your existing static HTML/JS setup stays the same
> - ✅ All inference runs privately in the user's browser
> - ✅ Real sklearn-trained models with real accuracy
> - The ONNX model files are ~10–300KB each (tiny)

> [!WARNING]
> **Drug & Diet Recommendations — No Real Dataset Exists**
>
> Your datasets cover **disease classification** (diabetes, heart disease, stroke, lung cancer) but there is **no drug prescription or diet dataset** in your project. For drug/diet recommendations, I have two options:
>
> **Option A (Recommended):** Keep the current rule-based engine for Stage 2 (drug/diet), but make Stage 1 (disease classification) use the real trained models. The drug/diet mapping stays the same — it's a clinical knowledge mapping, not ML prediction.
>
> **Option B:** Find and add a drug dataset (e.g., UCI Drug200 from Kaggle) for drug classification. This adds another model but requires downloading an additional dataset.
>
> **Which option do you prefer?**

---

## Proposed Changes

### Python Training Pipeline (new)

#### [NEW] `training/train_models.py`
A single Python script that:
1. Loads all 4 CSVs, cleans missing values, encodes categoricals
2. Trains models per disease:
   - **Diabetes:** Random Forest + DNN (keras) on 100K samples
   - **Heart Disease:** Random Forest + Gradient Boosting on Cleveland data
   - **Stroke:** Random Forest on stroke data
   - **Lung Cancer:** Random Forest on lung cancer data
3. Combines them into a **multi-disease ensemble** classifier
4. Computes **real feature importance** from trained RF models
5. Computes **real correlation matrix** from training data
6. Exports:
   - `models/disease_classifier.onnx` — ensemble ONNX model
   - `models/model_metadata.json` — feature names, class labels, scaler params, accuracy metrics
   - `js/data_computed.js` — real feature importance + correlation matrix (replaces hardcoded values)

#### [NEW] `training/requirements.txt`
```
scikit-learn
pandas
numpy
onnx
skl2onnx
```

---

### Frontend Changes

#### [MODIFY] `index.html`
- Add `<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>` to load ONNX runtime
- Update model architecture labels (show real architecture info)
- Show real accuracy metrics from trained models

#### [MODIFY] `js/ml-engine.js`
- Replace simulated `runDNN()`, `runRandomForest()`, `runGradientBoosting()` with a single `async runONNX(features)` method
- Load `disease_classifier.onnx` on first prediction call
- Preprocessing: normalize input features using stored scaler params from `model_metadata.json`
- Output: real probability distribution across disease classes
- Keep Stage 2 (drug/diet recommendation) rule-based (unless Option B chosen)

#### [MODIFY] `js/data.js`
- Replace hardcoded `FEATURE_IMPORTANCE` with real values computed from the trained model
- Replace hardcoded correlation matrix in charts.js with real computed values
- Add real model accuracy metrics (training accuracy, test accuracy, F1 scores)

#### [MODIFY] `js/charts.js`
- Update `renderCorrelationHeatmap` to use computed correlation matrix instead of hardcoded values
- Add new chart: model accuracy comparison (real test metrics)

#### [NEW] `models/disease_classifier.onnx`
The trained ONNX model file (~50–200KB)

#### [NEW] `models/model_metadata.json`
Scaler parameters, class mapping, feature names, accuracy metrics

---

## Open Questions

> [!IMPORTANT]
> 1. **Option A or B for drug/diet?** Keep rule-based (recommended) or add a drug dataset?
> 2. **Which diseases to keep?** Your current UI has 6 classes (T2DM, Hypertension, CVD, Metabolic Syndrome, Obesity, Healthy). With real data we have: Diabetes, Heart Disease, Stroke, Lung Cancer, Healthy. Should I map these to match your current UI or update the UI to reflect the real disease classes?
> 3. **EDA charts:** Should I compute real population statistics from the datasets for the distribution charts (replacing the synthetic data)?

---

## Verification Plan

### Automated Tests
1. Run `python training/train_models.py` — verify it completes without errors and produces ONNX + metadata files
2. Run the app in browser, load each sample patient, verify predictions match expected disease class
3. Check browser console for zero JS errors
4. Compare: same patient input → same output every time (determinism from ONNX)

### Manual Verification
- Load T2DM sample → should predict Diabetes with high confidence
- Load CVD sample → should predict Heart Disease with high confidence
- Load Healthy sample → should predict no disease / low risk scores
- Verify feature importance chart shows real computed values
- Verify correlation heatmap shows real computed values
