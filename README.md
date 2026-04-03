# ClinicalML — AI Drug & Diet Recommendation System

ClinicalML is an interactive, browser-native web application that simulates a two-stage machine learning pipeline for clinical trial enhancement and personalized patient care. 

In Stage 1, the system uses an ensemble of **real machine learning models** trained on over 100,000 patient records to predict one of seven disease classifications. In Stage 2, it acts as a recommendation engine to suggest personalized, clinically-focused diet and pharmacological interventions.

## 🌟 Key Features

*   **Zero-Dependency Browser Inference Engine:** Inference runs entirely locally via pure JavaScript (`js/ml-models.js`). The system executes pre-trained Random Forest, Gradient Boosting, and Multi-Layer Perceptron (DNN) mathematical models natively without any external backend servers or ONNX runtimes.
*   **Trained on Real Data:** Powered by `104,909` combined clinical records (Diabetes Prediction and Stroke Prediction public datasets).
*   **7-Class Disease Prediction:** Models are trained to differentiate and predict across 7 distinct clinical states:
    1. Type 2 Diabetes
    2. Hypertension
    3. Cardiovascular Disease (CVD)
    4. Stroke
    5. Metabolic Syndrome
    6. Obesity-related Disorder
    7. Healthy (Control)
*   **Dual Recommendation Engine:** Translates Stage 1 disease classifications and patient vitals into actionable, confidence-rated Diet and Drug schedules based on rule-based clinical guidelines.
*   **Data-Driven Exploratory Data Analysis (EDA):** The UI presents dynamic Radar charts, interactive Feature Importance graphs, and Correlation Heatmaps computed directly from the real-world statistical distribution of the training data.

## 📂 Project Structure

```
c:\Users\Lenovo\Drug_project\
├── index.html               # Main application and UI layout
├── login.html               # Login page for authentication
├── firebase.json            # Firebase hosting configuration
├── .firebase/               # Firebase local setup
├── .venv/                   # Python virtual environment
├── datasets/                # CSV datasets for training containing real world data
├── css/
│   └── index.css            # Modern glassmorphism design system
├── js/
│   ├── ui.js                # Form handling, UI navigation, styling states
│   ├── charts.js            # Chart.js integraton for EDA and Results graphics
│   ├── data.js              # Reference lists & metadata (Drugs, Diets, Classes)
│   ├── data_computed.js     # Real dataset statistics (correlation matrix, feature dist)
│   ├── ml-models.js         # Core in-browser ML inference logic (tree traversal, MLP)
│   └── ml-engine.js         # Final ensemble voting + Stage 2 recommendation logic
├── models/                  # Exported JSON formats of the trained models
│   ├── rf_model.json
│   ├── gb_model.json
│   ├── dnn_weights.json
│   └── metadata.json
└── training/                # Python scripts for data processing and model training
    ├── train_models.py      # ML pipeline to ingest CSVs and export models to JSON
    └── requirements.txt     # Python requirements (scikit-learn, pandas, numpy)
```

## 🌐 Live URLs

The application is deployed and live on the internet!
- **GitHub Pages**: [https://palleviswanathreddy-max.github.io/ClinicalML-project/](https://palleviswanathreddy-max.github.io/ClinicalML-project/)
- **Vercel**: Live and automatically synced (check your Vercel dashboard for the exact URL).

## 🚀 How to PUSH and DEPLOY Updates

We have configured a fully automated CI/CD pipeline. Whenever you modify the application code or data:
1. Double click the **`deploy.bat`** file from your file explorer (or simply run `.\deploy.bat` in the terminal).
2. Enter a commit message describing what you changed.
3. The file will automatically commit everything to GitHub. This triggers an immediate, automatic re-deployment to your live **Vercel** and **GitHub Pages** URLs.

## 💻 How to Run the Application Locally

Because ClinicalML uses lightweight JSON representations of models natively in-browser, you do not need a backend API. 

### Run with Python (Standard)
1. Navigate into the project directory in your terminal.
2. Start the python server: `python -m http.server 3030`
3. Open your browser and navigate to: `http://localhost:3030`

### Run with Node.js
If you prefer NPM:
1. Open terminal in the project directory.
2. Run: `npx serve . --listen 3030`
3. Open your browser and navigate to: `http://localhost:3030`

## 🧠 Retraining the Machine Learning Models

If you wish to augment the initial data or modify the model architecture, you can retrain the models:

1. Ensure you have the raw CSV clinical data in the `datasets/` directory:
   - `diabetes_prediction_dataset.csv`
   - `healthcare-dataset-stroke-data.csv`
   - `heart_cleveland_upload.csv`
2. Navigate to the `training` directory and install the requirements (a virtual environment like `.venv` is recommended):
   ```bash
   cd training
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train_models.py
   ```
   *The script will automatically parse the data, handle class balancing, train the models, compute feature statistics, and overwrite the JSON models in `/models` and `/js/data_computed.js`.*

## 💻 Tech Stack

*   **Frontend UI:** Vanilla HTML, CSS, JavaScript
*   **Visualizations:** Chart.js
*   **Machine Learning (Training):** Python, Scikit-Learn, Pandas, NumPy
*   **Machine Learning (Inference):** Pure JavaScript JSON ingestion
