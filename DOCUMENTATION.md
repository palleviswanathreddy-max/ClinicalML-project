# ClinicalML: Comprehensive System Architecture & Technical Documentation

## Table of Contents
1. Executive Summary
2. Introduction & Clinical Background
3. System Architecture Overview
4. Data Acquisition & Preprocessing Strategy
5. Stage 1: Machine Learning Training Pipeline
6. Model Exportation Strategy (Zero-Dependency)
7. Browser-Native Inference Engine
8. Stage 2: Clinical Recommendation Engine
9. Frontend Architecture & Exploratory Data Analysis
10. Deployment & Hosting DevOps
11. Developer Guide & API Reference

---

## 1. Executive Summary

ClinicalML is an interactive, browser-native web application designed to act as a two-stage machine learning pipeline and recommendation system for clinical trial enhancement and personalized patient care. The application eliminates the traditional requirements for a heavy backend server (such as Flask, FastAPI, or ONNX runtimes) by executing mathematical inference entirely in pure JavaScript within the client's browser. 

The system leverages real-world clinical datasets comprising over 100,000 patient records to predict seven distinct clinical states: Type 2 Diabetes, Hypertension, Cardiovascular Disease (CVD), Stroke, Metabolic Syndrome, Obesity-related Disorder, and Healthy/Control. 

By utilizing an ensemble consisting of Random Forest, Gradient Boosting, and a Multi-Layer Perceptron (DNN), the system translates clinical biomarkers into a rigorous, probability-scored diagnosis. In its final step, a rule-based expert recommendation engine parses these predictions to deliver actionable dietary and pharmacological interventions.

## 2. Introduction & Clinical Background

### 2.1 The Need for Localized AI in Healthcare
In modern clinical informatics, deploying predictive models usually involves sending sensitive Patient Health Information (PHI) over the network via RESTful APIs to remote cloud GPU clusters. While scalable, this approach violates zero-trust policies and poses severe GDPR/HIPAA compliance challenges.

ClinicalML solves this by implementing **Edge Inference**. The mathematical weights and topological structures computed during the Python training phase are meticulously compiled into flat, highly compressed JSON formats. These static JSON files are downloaded once by the client. The browser's native V8 JavaScript engine then rebuilds the matrices and tree nodes in memory. Consequently, patient data *never* leaves the user's browser.

### 2.2 The 7-Class Disease Taxonomy
The model is specifically trained to solve a multi-class categorization problem with the following discrete clinical states:

1. **Type 2 Diabetes (t2dm):** Flagged in patients with chronically high blood glucose levels (Fast Plasma Glucose) combined with other metabolic risk factors.
2. **Hypertension:** High blood pressure acting as a precursor to CVD or stroke.
3. **Cardiovascular Disease (CVD):** Broad category encompassing heart disease histories.
4. **Stroke:** Predicting the likelihood of severe cerebral events based on age, BMI, and glucose combinations.
5. **Metabolic Syndrome:** A composite label applied to patients exhibiting simultaneous hypertension, elevated glucose, and obesity.
6. **Obesity-related Disorder:** Triggered mathematically when BMI heavily exceeds safe thresholds (e.g., > 30 kg/m²).
7. **Healthy (Control):** The absence of the primary indicators.

---

## 3. System Architecture Overview

The system architecture is bifurcated into an **Offline Training Pipeline** and an **Online Inference Engine**. 

### 3.1 Offline Architecture
- **Data Layer:** Merges raw CSV schemas (`diabetes_prediction_dataset.csv`, `healthcare-dataset-stroke-data.csv`, `heart_cleveland_upload.csv`).
- **Python ML Pipeline:** Uses `scikit-learn` and `pandas` to harmonize columns, handle class imbalance via capping and oversampling, scale numerical data features, and fit three independent models.
- **Serialization Layer:** Custom Python functions trace the binary tree nodes of Random Forests and Gradient Boosters, extracting leaf probabilities and decision boundaries to dump into static `.json` files.

### 3.2 Online Architecture
- **Static Delivery:** Firebase Hosting serves `index.html`, vanilla CSS, and the pre-computed JSON weights.
- **Client-Side Engine:** `ml-models.js` intercepts user form inputs, applies the exact Standard Scaler transformations, and traverses the parsed JSON graphs natively.
- **Stage 2 Recommender:** A deterministic engine evaluates the winning diagnostic class and maps it to evidence-based drug schedules (e.g., Metformin for T2DM) and diet profiles (e.g., DASH diet for Hypertension).
- **Presentation:** Chart.js instantly renders Feature Importance arrays and Correlation Matrices dynamically computed from the static datasets.

---

## 4. Data Acquisition & Preprocessing Strategy

Machine learning models are only as robust as the data they consume. ClinicalML aggregates multiple distinct datasets.

### 4.1 Datasets
1. **Diabetes Prediction Dataset (100,000 rows):** Contains features like `age`, `gender`, `bmi`, `hypertension`, `heart_disease`, `smoking_history`, and `blood_glucose_level`.
2. **Stroke Dataset (5,110 rows):** Contains analogous demographic information but incorporates `avg_glucose_level` and a discrete `stroke` binary output.
3. **Heart Disease Dataset:** Provides supplementary data points for secondary cardiovascular validation.

### 4.2 Data Harmonization & Feature Engineering
Because the original datasets were recorded in different formats, a rigid harmonization script (`train_models.py`) performs the following mappings:

- **Smoking History:** Categorical text like "never", "formerly smoked", and "smokes" are mapped to a generic ordinal scale (`0 = never`, `1 = former`, `2 = current`).
- **Blood Glucose:** `avg_glucose_level` from Stroke data and `blood_glucose_level` from Diabetes data are coerced to universally represent the continuous variable `blood_glucose`.
- **Gender:** Mapped to binary categorical indices (`Male=1`, `Female=0`).

### 4.3 Target Label Construction
To combine datasets that predict solitary outcomes (one predicts diabetes, the other stroke) into a unified 7-class prediction problem, the pipeline employs a deterministic priority assignment rule (`assign_disease_class()`):
1. **Stroke** prediction takes supreme priority if history indicates it.
2. **CVD** takes secondary priority.
3. **T2DM** takes tertiary priority.
4. If no severe event is logged, but the patient has high hypertension and high BMI + Glucose, the label falls back to **Metabolic**.
5. Failing that, **Obesity** is evaluated based purely on BMI. 

### 4.4 Class Imbalance Optimization
The "Healthy" class drastically outnumbered clinical events in the raw demographics. Without balancing, the neural network would simply predict "Healthy" 100% of the time and achieve ~90% raw accuracy.
- **Undersampling:** The healthy class and heavily skewed classes are randomly sampled down to a cap limit of 5,000 records using `df.sample(n=target_size)`.
- **Oversampling:** Scarce classes (such as Metabolic cases) are artificially bootstrapped up to 1,000 records using random sampling with replacement (`replace=True`).

---

## 5. Stage 1: Machine Learning Training Pipeline

The offline core executed by `python training/train_models.py`.

### 5.1 Random Forest Classifier (RF)
The pipeline fits a Bagged Ensemble using `sklearn.ensemble.RandomForestClassifier`.
- **Parameters:** `n_estimators=30`, `max_depth=8`, `min_samples_split=10`.
- **Rationale:** Random forest inherently handles non-linear relationships in tabular clinical data well, mapping decision boundaries like (`glucose > 180 AND BMI > 28`). We strictly limit the depth to 8 to prevent the resulting JSON file size from ballooning into multi-megabyte payloads, ensuring web-friendly deliverability.

### 5.2 Gradient Boosting Classifier (GB)
- **Parameters:** `n_estimators=80`, `max_depth=4`, `learning_rate=0.1`.
- **Rationale:** GB builds sequential trees that correct the residuals of preceding trees. To accommodate multi-class classification, `scikit-learn` implements this as a One-vs-Rest strategy. This means for *K* classes, there are actually *K × n_estimators* distinct regression trees built (i.e., 7 × 80 = 560 miniature decision trees).

### 5.3 Multi-Layer Perceptron (DNN)
- **Parameters:** `hidden_layer_sizes=(64, 32, 16)`, `activation='relu'`, `solver='adam'`.
- **Rationale:** A Deep Neural Network is introduced to capture hidden interactive feature combinations (like how Age, Smoking, and BMI compound non-linearly combined). The network takes 7 inputs (the 7 clinical features), pushes them through three ReLU-activated hidden layers, and outputs a 7-node softmax array representing class probabilities.

### 5.4 Standardization
Vital to distance-based metric calculations (like MLP gradient descent), the 7-dimensional input arrays `(X)` are processed through a `StandardScaler`.
```python
X_scaled = (X - scaler.mean_) / scaler.scale_
```
These parameters (`mean_` and `scale_`) are extracted and serialized so the JS engine can mirror the transformation exactly.

---

## 6. Model Exportation Strategy (Zero-Dependency)

This section details the critical engineering innovation that enables serverless AI routing. 

### 6.1 Exporting the Random Forest
A decision tree in `scikit-learn` is stored as parallel arrays (`children_left`, `children_right`, `feature`, `threshold`). We traverse these arrays recursively in Python to dump out dictionary equivalents:
```json
{
  "f": 3,           // Split on Feature index 3 (glucose)
  "t": 125.5,       // Threshold
  "l": 5,           // Go to node 5 if <=
  "r": 12           // Go to node 12 if >
}
```
Leaf nodes contain normalized label probability arrays:
```json
{"leaf": [0.01, 0.90, ..., 0.0]}
```

### 6.2 Exporting Gradient Boosting
GB trees don't output probabilities; they output logic scores. They also have an initial prior (`init_scores`). The JSON payload stores:
1. The `learning_rate` scalar.
2. The initial log-odds arrays.
3. 560 sequential trees where leaf nodes output raw summation scalars.

### 6.3 Exporting the Neural Network
The MLP weights and biases matrices (`mlp.coefs_` and `mlp.intercepts_`) are rounded to 5 decimal places to save string space and dumped into lists of lists:
```json
{
  "layers": [
    {
      "weights": [[0.51, ...], [-0.14, ...]],
      "biases": [0.04, ...],
      "activation": "relu"
    }
  ]
}
```

---

## 7. Browser-Native Inference Engine

In the browser UI layer, `js/ml-models.js` is responsible for fetching these JSONs via native HTTP `fetch` APIs and managing the execution context seamlessly.

### 7.1 Reconstructed Feature Scaling
Before inference can run, the exact user inputs gathered via HTML forms are mapped into a contiguous 7-element float array, matching the exact column order expected by the models:
`[age, gender_enc, bmi, blood_glucose, hypertension, heart_disease_hx, smoking_enc]`

It processes this vector using the `mean` and `scale` metadata extracted during the offline phase:
```javascript
const scaledInputs = inputs.map((val, i) => (val - mean[i]) / scale[i]);
```

### 7.2 Native Predict: Random Forest
The JS engine simulates standard tree traversal using a basic integer `while` loop that resolves node boundaries:
```javascript
let nodeIndex = 0;
while (!trees[nodeIndex].leaf) {
  const node = trees[nodeIndex];
  if (scaledInputs[node.f] <= node.t) {
    nodeIndex = node.l;
  } else {
    nodeIndex = node.r;
  }
}
// Returns the leaf probability array.
```
This loop executes in fractions of a microsecond inside the browser's fast JIT compiler, achieving millions of inferences per second locally.

### 7.3 Native Predict: Gradient Boosting
For GB models, the inference must evaluate `N` regression trees per class, multiply leaf outcomes by the learning rate, sum against the initial probabilities, and then push the cumulative score through an exponential Softmax mechanism to yield true probabilities. The JavaScript reconstructs this linear combination process accurately.

### 7.4 Native Predict: Neural Networks (MLP)
The Deep Neural Network inference requires custom linear algebra subroutines written natively in vanilla JS:
1. Calculate dot-products `(Inputs • Weights + Biases)`.
2. Apply `ReLU(x) = Math.max(0, x)` activation dynamically.
3. Forward to the final layer.
4. Calculate final activation function `Softmax`.

### 7.5 The Voting Ensemble mechanism
Once all three distinct models independently calculate 7 predicted probabilities, an `Ensemble Vote` triggers. Each model casts an integer 'vote' for the class with its highest max probability. The mode (most frequent item) of this 3-vote array is selected as the definitive ground-truth clinical diagnosis, ensuring that anomalies generated by an individual decision tree are statistically suppressed by the other ML protocols.

---

## 8. Stage 2: Clinical Recommendation Engine

The ML Model stops at simply providing a diagnostic term (e.g. `t2dm`). Stage 2 invokes the logic located within `js/ml-engine.js` and `js/data.js` to translate that tag into human-readable action.

### 8.1 Pharmacological Guideline Router
Based on the winning label, specific standard-of-care FDA/EMA guidelines are triggered.
- If `t2dm`, the engine recommends: "Metformin ER 500mg (Daily-Initial)".
- If `stroke`, the engine highlights: "Clopidogrel (Plavix) 75mg maintenance" and orders urgent consultations based on acute protocols.
- If `cvd`, it may suggest: "Atorvastatin 80mg (High-Intensity)".

### 8.2 Dietary Intervention Protocol
Alongside chemical interventions, the recommendation engine selects targeted structured diets.
- **DASH Diet:** Suggested for hypertension triggers, emphasizing high potassium vegetables, low sodium, and lean meats.
- **Mediterranean Diet:** Highlighted heavily for CVD/Stroke warnings pointing to omega-3 necessity.
- **Low-GI Profiles:** Mandated for T2DM or Metabolic syndrome to mitigate insulin spike conditions.

These dual recommendations act as "Clinical Decision Support Systems" (CDSS) for researchers exploring personalized protocol allocations in clinical trials.

---

## 9. Frontend Architecture & Exploratory Data Analysis

The frontend of ClinicalML prioritizes rich aesthetics, vibrant visualizations, and fluid glassmorphism utilizing Vanilla CSS without a bulky dependency load from React or Angular.

### 9.1 The Glassmorphism UI
Located within `index.css`, modern CSS features utilize heavy backdrop filters to construct floating, semi-transparent frosted panels:
```css
.frosted-glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}
```
Combined with dark mesh gradients and high-contrast Google Fonts like **Inter**, these UI elements assure the product has an immensely premium, state-of-the-art feel, elevating user trust and cognitive comfort during dense data analysis.

### 9.2 Real-time Exploratory Data Analysis (EDA)
In a typical machine learning dashboard, plotting a correlation matrix involves querying an R/Python server to render a PNG image. Alternatively, ClinicalML uses `data_computed.js` which was statically produced during the original training script context.
This script embeds the literal `6x6` mathematical correlation matrix and the distribution arrays directly as plain Javascript constants. The application utilizes the robust `Chart.js` engine within `js/charts.js` to draw interactive visual structures.

- **Radar Charts:** Draw the normalized patient profile parameters across clinical axes in real time.
- **Feature Importance:** Rendered as horizontal bar charts indicating that `hypertension history` typically commands a 0.28 coefficient weighting whereas `age` commands a heavy 0.32 impact for T2DM.
- **Heatmaps and Demographics Distribution Line Charts:** Visualize the original standard deviations to give patients and doctors context of how heavily the patient deviates from the generalized population curve.

---

## 10. Deployment & Hosting DevOps

### 10.1 Static Asset Structure
Because there's no backend state, the compiled output comprises exclusively `.html`, `.css`, `.js`, and `.json` assets. This inherently guarantees un-hackable server states (no SQL Injection or Remote Code Execution vectors exist at run time) and delivers ultimate scaling throughput constraints.

### 10.2 Firebase Hosting
The project relies on Firebase native hosting features. Configured securely via `firebase.json`:
- **Cache Invalidations:** Implements crucial custom headers. Due to the rapid iteration cycle of ML models (often updating `rf_model.json`), it is imperative that client browser caches do not stubbornly cache stale models.
- **`Cache-Control: no-cache, no-store, must-revalidate`** is strictly enforced on all JSON files so every page load forces immediate validation of the mathematical network weights.
- The `firebase.json` automatically ignores `.venv` directories and massive raw dataset folders, pushing solely the compiled, optimized frontend.

---

## 11. Developer Guide & API Reference

### 11.1 Project Hierarchy
* `/css/` - Stores the core `index.css` stylesheet for the entire application.
* `/js/` - Houses functional logic.
  * `ml-models.js` is the offline inference mechanism.
  * `ui.js` processes HTML DOM manipulations and tab route handling.
  * `charts.js` abstracts all the long-form Chart.Js configurations into clean functions.
* `/models/` - The compiled JSON directory holding the model representations.
* `/training/` - Python utilities explicitly for standardizing CSV inputs and outputting valid JSON trees.
* `/datasets/` - Safe repository meant to hold raw PII/Healthcare Data not checked into Firebase hosting.

### 11.2 Re-training execution pipeline
To expand the system’s predictive power effectively (e.g., teaching it to analyze "Liver Failure Parameters"), follow to instructions below:
1. Accumulate your clinical history data as CSV files and migrate them into the `/datasets` root.
2. Edit `train_models.py` in the `/training` directory:
   - Introduce a new class label within the `CLASS_NAMES` variables list.
   - Inject priority logic into `assign_disease_class()`.
3. Open a shell and configure the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # (or activate.bat natively on Windows)
   pip install -r requirements.txt
   ```
4. Execute the training protocol:
   ```bash
   python train_models.py
   ```
Upon conclusion, the script mathematically rewrites `/models/rf_model.json` and updates `js/data_computed.js`. If you maintain `localhost:3030` via `npx serve`, simply hit refresh in your browser. The web application dynamically assimilates the freshly engineered neural weights context and is instantaneously deployed to the edge.

> [!TIP]
> **Extending Feature Inputs:** If you add more clinical inputs (e.g., adding `cholesterol_levels` as input 8), you must update `FEATURE_NAMES` in Python, add an input field `<input id="cholesterol">` into the `index.html` structure, and modify `ui.js` data-grabbing array structure to match the feature list dimension exactly (extending the array input length to 8).

---

*(End of Comprehensive Technical Documentation)*
