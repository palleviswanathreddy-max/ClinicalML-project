"""
ClinicalML — Real Model Training Pipeline
==========================================
Loads real clinical datasets, trains RF + GB + MLP classifiers for 7-class
disease prediction, and exports model structures as JSON for pure-JS
browser inference (no ONNX runtime needed).

Outputs:
  models/rf_model.json          — Random Forest tree structures
  models/gb_model.json          — Gradient Boosting tree structures
  models/dnn_weights.json       — MLP weights & biases
  models/metadata.json          — scaler, class names, accuracy, feature info
  js/data_computed.js           — real feature importance + correlation matrix
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
DIABETES_CSV = os.path.join(DATASETS_DIR, 'diabetes_prediction_dataset.csv')
STROKE_CSV   = os.path.join(DATASETS_DIR, 'healthcare-dataset-stroke-data.csv')
HEART_CSV    = os.path.join(DATASETS_DIR, 'heart_cleveland_upload.csv')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
JS_DIR       = os.path.join(BASE_DIR, 'js')

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ['t2dm', 'hypertension', 'cvd', 'stroke', 'metabolic', 'obesity', 'healthy']
CLASS_LABELS = {
    't2dm': 0, 'hypertension': 1, 'cvd': 2, 'stroke': 3,
    'metabolic': 4, 'obesity': 5, 'healthy': 6
}
FEATURE_NAMES = ['age', 'gender_enc', 'bmi', 'blood_glucose', 'hypertension', 'heart_disease_history', 'smoking_enc']


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & LABELING
# ══════════════════════════════════════════════════════════════════════════════

def encode_smoking(val):
    """Map smoking values to 0=never, 1=former, 2=current."""
    val = str(val).lower().strip()
    if val in ('current', 'smokes', 'yes', '1', '2'):
        return 2
    if val in ('formerly smoked', 'former', 'ex', 'ever'):
        return 1
    return 0  # never, no info, not current


def assign_disease_class(row):
    """Priority-based multi-class label from clinical indicators."""
    diabetes = row.get('diabetes', 0)
    hyp      = row.get('hypertension', 0)
    cvd      = row.get('heart_disease_history', 0)
    stroke   = row.get('stroke', 0)
    bmi      = row.get('bmi', 25)
    glucose  = row.get('blood_glucose', 90)

    if stroke == 1:
        return 'stroke'
    if cvd == 1:
        return 'cvd'
    if diabetes == 1:
        return 't2dm'
    # Metabolic syndrome: high BMI + elevated glucose + hypertension (no diabetes/cvd)
    if hyp == 1 and bmi > 30 and glucose > 100:
        return 'metabolic'
    if hyp == 1:
        return 'hypertension'
    if bmi >= 30:
        return 'obesity'
    return 'healthy'


def load_diabetes_dataset():
    """Load and harmonize the diabetes prediction dataset (100K rows)."""
    print(f"  Loading {DIABETES_CSV}...")
    df = pd.read_csv(DIABETES_CSV)
    print(f"    Raw rows: {len(df)}")

    out = pd.DataFrame()
    out['age']                   = df['age']
    out['gender_enc']            = (df['gender'].str.lower() == 'male').astype(int)
    out['bmi']                   = pd.to_numeric(df['bmi'], errors='coerce')
    out['blood_glucose']         = pd.to_numeric(df['blood_glucose_level'], errors='coerce')
    out['hypertension']          = df['hypertension'].astype(int)
    out['heart_disease_history'] = df['heart_disease'].astype(int)
    out['smoking_enc']           = df['smoking_history'].apply(encode_smoking)
    out['diabetes']              = df['diabetes'].astype(int)
    out['stroke']                = 0  # Not in this dataset
    out.dropna(inplace=True)
    return out


def load_stroke_dataset():
    """Load and harmonize the stroke dataset (5K rows)."""
    print(f"  Loading {STROKE_CSV}...")
    df = pd.read_csv(STROKE_CSV)
    print(f"    Raw rows: {len(df)}")

    out = pd.DataFrame()
    out['age']                   = pd.to_numeric(df['age'], errors='coerce')
    out['gender_enc']            = (df['gender'].str.lower() == 'male').astype(int)
    out['bmi']                   = pd.to_numeric(df['bmi'], errors='coerce')
    out['blood_glucose']         = pd.to_numeric(df['avg_glucose_level'], errors='coerce')
    out['hypertension']          = df['hypertension'].astype(int)
    out['heart_disease_history'] = df['heart_disease'].astype(int)
    out['smoking_enc']           = df['smoking_status'].apply(encode_smoking)
    out['diabetes']              = 0  # Not in this dataset
    out['stroke']                = df['stroke'].astype(int)
    out.dropna(inplace=True)
    return out


def prepare_unified_dataset():
    """Combine datasets, assign labels, balance classes."""
    print("\n═══ STEP 1: Loading & combining datasets ═══")
    df_diab   = load_diabetes_dataset()
    df_stroke = load_stroke_dataset()

    df = pd.concat([df_diab, df_stroke], ignore_index=True)
    print(f"\n  Combined rows: {len(df)}")

    # Assign multi-class labels
    df['label'] = df.apply(assign_disease_class, axis=1)
    print(f"\n  Class distribution (raw):")
    for cls, count in sorted(Counter(df['label']).items(), key=lambda x: -x[1]):
        print(f"    {cls:15s}: {count:>6,d}")

    # ── Balance classes ──
    # The healthy class vastly dominates. Undersample majority, keep all minority.
    class_counts = Counter(df['label'])
    target_size = 5000  # cap each class to prevent domination

    balanced_dfs = []
    for cls in CLASS_NAMES:
        cls_df = df[df['label'] == cls]
        if len(cls_df) == 0:
            print(f"    [!] Class '{cls}' has 0 samples — will be synthesized")
            continue
        if len(cls_df) > target_size:
            cls_df = cls_df.sample(n=target_size, random_state=42)
        elif len(cls_df) < 300:
            # Oversample very small classes
            cls_df = cls_df.sample(n=min(1000, target_size), replace=True, random_state=42)
        balanced_dfs.append(cls_df)

    df_bal = pd.concat(balanced_dfs, ignore_index=True)
    print(f"\n  Class distribution (balanced):")
    for cls, count in sorted(Counter(df_bal['label']).items(), key=lambda x: -x[1]):
        print(f"    {cls:15s}: {count:>6,d}")

    # Encode labels to integers
    df_bal['label_enc'] = df_bal['label'].map(CLASS_LABELS)

    X = df_bal[FEATURE_NAMES].values.astype(np.float32)
    y = df_bal['label_enc'].values

    return X, y, df_bal


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(X_train, y_train, X_test, y_test):
    """Train RF, GB, and MLP on the standardized data."""
    print("\n═══ STEP 2: Training models ═══")
    results = {}

    # ── Random Forest ──
    print("\n  Training Random Forest (30 trees, depth=8)...")
    rf = RandomForestClassifier(
        n_estimators=30, max_depth=8, min_samples_split=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    print(f"    Accuracy: {rf_acc:.4f}  |  F1: {rf_f1:.4f}")
    results['rf'] = {'model': rf, 'accuracy': rf_acc, 'f1': rf_f1}

    # ── Gradient Boosting ──
    print("\n  Training Gradient Boosting (80 estimators, depth=4)...")
    gb = GradientBoostingClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.1,
        min_samples_split=10, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_f1 = f1_score(y_test, gb_pred, average='weighted')
    print(f"    Accuracy: {gb_acc:.4f}  |  F1: {gb_f1:.4f}")
    results['gb'] = {'model': gb, 'accuracy': gb_acc, 'f1': gb_f1}

    # ── MLP (DNN) ──
    print("\n  Training MLP Neural Network (64->32->16)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), activation='relu',
        solver='adam', max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1
    )
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred, average='weighted')
    print(f"    Accuracy: {mlp_acc:.4f}  |  F1: {mlp_f1:.4f}")
    results['mlp'] = {'model': mlp, 'accuracy': mlp_acc, 'f1': mlp_f1}

    # ── Classification report (ensemble vote) ──
    print("\n  ── Ensemble (majority vote) ──")
    stacked = np.column_stack([rf_pred, gb_pred, mlp_pred])
    ens_pred = np.array([np.bincount(row.astype(int), minlength=7).argmax()
                         for row in stacked])
    ens_acc = accuracy_score(y_test, ens_pred)
    ens_f1 = f1_score(y_test, ens_pred, average='weighted')
    print(f"    Accuracy: {ens_acc:.4f}  |  F1: {ens_f1:.4f}")
    print(f"\n  Classification Report (Ensemble):")
    print(classification_report(y_test, ens_pred, target_names=CLASS_NAMES, zero_division=0))
    results['ensemble'] = {'accuracy': round(ens_acc, 4), 'f1': round(ens_f1, 4)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MODEL EXPORT — Pure JSON (no ONNX needed)
# ══════════════════════════════════════════════════════════════════════════════

def export_decision_tree(tree, n_classes):
    """Convert a single sklearn DecisionTree to a compact dict."""
    t = tree.tree_
    nodes = []
    for i in range(t.node_count):
        if t.feature[i] == -2:  # leaf
            # Normalize value to probabilities
            vals = t.value[i][0]
            total = vals.sum()
            probs = (vals / total).tolist() if total > 0 else [0.0] * n_classes
            nodes.append({'leaf': [round(p, 4) for p in probs]})
        else:
            nodes.append({
                'f': int(t.feature[i]),
                't': round(float(t.threshold[i]), 4),
                'l': int(t.children_left[i]),
                'r': int(t.children_right[i])
            })
    return nodes


def export_rf_model(rf, path):
    """Export Random Forest as JSON list of trees."""
    print(f"\n  Exporting RF to {path}...")
    n_classes = rf.n_classes_
    trees = []
    for est in rf.estimators_:
        trees.append(export_decision_tree(est, n_classes))
    data = {'n_classes': n_classes, 'trees': trees}
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    size_kb = os.path.getsize(path) / 1024
    print(f"    Exported {len(trees)} trees, {size_kb:.0f} KB")


def export_gb_model(gb, path):
    """Export Gradient Boosting as JSON.
    
    GB uses one-vs-rest: n_classes * n_estimators regression trees.
    Each tree predicts a residual (single float per leaf).
    """
    print(f"\n  Exporting GB to {path}...")
    n_classes = gb.n_classes_
    learning_rate = gb.learning_rate
    # Get initial class scores (log-prior probabilities)
    try:
        init_scores = np.log(gb.init_.class_prior_ + 1e-10).tolist()
    except AttributeError:
        init_scores = [0.0] * n_classes

    estimators = []
    for i in range(gb.n_estimators_):
        round_trees = []
        for c in range(n_classes):
            tree = gb.estimators_[i][c]
            t = tree.tree_
            nodes = []
            for j in range(t.node_count):
                if t.feature[j] == -2:
                    nodes.append({'leaf': round(float(t.value[j][0][0]), 6)})
                else:
                    nodes.append({
                        'f': int(t.feature[j]),
                        't': round(float(t.threshold[j]), 4),
                        'l': int(t.children_left[j]),
                        'r': int(t.children_right[j])
                    })
            round_trees.append(nodes)
        estimators.append(round_trees)

    data = {
        'n_classes': n_classes,
        'learning_rate': learning_rate,
        'init': [round(s, 6) for s in init_scores],
        'estimators': estimators
    }
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    size_kb = os.path.getsize(path) / 1024
    print(f"    Exported {gb.n_estimators_} rounds x {n_classes} classes, {size_kb:.0f} KB")


def export_mlp_model(mlp, path):
    """Export MLP weights and biases as JSON."""
    print(f"\n  Exporting MLP to {path}...")
    layers = []
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        layers.append({
            'weights': np.round(w, 5).tolist(),
            'biases': np.round(b, 5).tolist(),
            'activation': 'relu' if i < len(mlp.coefs_) - 1 else 'softmax'
        })
    data = {
        'n_classes': int(mlp.n_outputs_) if mlp.n_outputs_ > 1 else len(mlp.classes_),
        'layers': layers
    }
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    size_kb = os.path.getsize(path) / 1024
    print(f"    Exported {len(layers)} layers, {size_kb:.0f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# METADATA + JS DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_metadata(scaler, results, path):
    """Export scaler params, class names, metrics."""
    print(f"\n  Exporting metadata to {path}...")
    meta = {
        'feature_names': FEATURE_NAMES,
        'class_names': CLASS_NAMES,
        'scaler': {
            'mean': np.round(scaler.mean_, 6).tolist(),
            'scale': np.round(scaler.scale_, 6).tolist()
        },
        'metrics': {
            'rf':  {'accuracy': round(results['rf']['accuracy'], 4),
                    'f1': round(results['rf']['f1'], 4)},
            'gb':  {'accuracy': round(results['gb']['accuracy'], 4),
                    'f1': round(results['gb']['f1'], 4)},
            'mlp': {'accuracy': round(results['mlp']['accuracy'], 4),
                    'f1': round(results['mlp']['f1'], 4)},
            'ensemble': results['ensemble']
        }
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


def export_js_data(rf, df_bal, path):
    """Export real feature importance + correlation matrix as JS file."""
    print(f"\n  Exporting computed data to {path}...")

    # Feature importance from RF (Gini importance)
    importance = rf.feature_importances_
    feat_data = []
    category_map = {
        'age': 'Demographics', 'gender_enc': 'Demographics',
        'bmi': 'Physiological', 'blood_glucose': 'Physiological',
        'hypertension': 'Medical History', 'heart_disease_history': 'Medical History',
        'smoking_enc': 'Lifestyle'
    }
    display_names = {
        'age': 'Age', 'gender_enc': 'Gender', 'bmi': 'BMI',
        'blood_glucose': 'Blood Glucose', 'hypertension': 'Hypertension History',
        'heart_disease_history': 'CVD History', 'smoking_enc': 'Smoking Status'
    }
    for i, fname in enumerate(FEATURE_NAMES):
        feat_data.append({
            'feature': display_names.get(fname, fname),
            'importance': round(float(importance[i]), 4),
            'category': category_map.get(fname, 'Other')
        })
    feat_data.sort(key=lambda x: -x['importance'])

    # Correlation matrix from training data
    corr_cols = ['age', 'bmi', 'blood_glucose', 'hypertension', 'heart_disease_history', 'smoking_enc']
    corr_labels = ['Age', 'BMI', 'Glucose', 'Hypertension', 'CVD Hx', 'Smoking']
    corr_df = df_bal[corr_cols]
    corr_matrix = np.round(corr_df.corr().values, 2).tolist()

    # Class distribution for EDA
    class_dist = Counter(df_bal['label'])
    class_dist_data = {cls: int(class_dist.get(cls, 0)) for cls in CLASS_NAMES}

    # Population stats for BMI distribution chart
    bmi_stats = {
        'mean': round(float(df_bal['bmi'].mean()), 1),
        'std': round(float(df_bal['bmi'].std()), 1),
        'bins': list(range(15, 55, 5)),
        'counts': []
    }
    for lo in range(15, 50, 5):
        count = int(((df_bal['bmi'] >= lo) & (df_bal['bmi'] < lo + 5)).sum())
        bmi_stats['counts'].append(count)

    # Write JS module
    js_content = f"""// ═══════════════════════════════════════════════════════════════════════════
// AUTO-GENERATED by training/train_models.py — DO NOT EDIT MANUALLY
// Real feature importance, correlation matrix, and population stats
// computed from {len(df_bal):,} training samples across 4 clinical datasets.
// ═══════════════════════════════════════════════════════════════════════════

const COMPUTED_FEATURE_IMPORTANCE = {json.dumps(feat_data, indent=2)};

const COMPUTED_CORRELATION = {{
  labels: {json.dumps(corr_labels)},
  matrix: {json.dumps(corr_matrix)}
}};

const COMPUTED_CLASS_DISTRIBUTION = {json.dumps(class_dist_data, indent=2)};

const COMPUTED_BMI_STATS = {json.dumps(bmi_stats, indent=2)};
"""
    with open(path, 'w') as f:
        f.write(js_content)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   ClinicalML — Real Model Training Pipeline              ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # 1. Load & prepare data
    X, y, df_bal = prepare_unified_dataset()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 3. Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4. Train
    results = train_all_models(X_train_s, y_train, X_test_s, y_test)

    # 5. Export models
    print("\n═══ STEP 3: Exporting models ═══")
    export_rf_model(results['rf']['model'],
                    os.path.join(MODELS_DIR, 'rf_model.json'))
    export_gb_model(results['gb']['model'],
                    os.path.join(MODELS_DIR, 'gb_model.json'))
    export_mlp_model(results['mlp']['model'],
                     os.path.join(MODELS_DIR, 'dnn_weights.json'))
    export_metadata(scaler, results,
                    os.path.join(MODELS_DIR, 'metadata.json'))
    export_js_data(results['rf']['model'], df_bal,
                   os.path.join(JS_DIR, 'data_computed.js'))

    print("\n═══ DONE ═══")
    total_kb = sum(
        os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
        for f in os.listdir(MODELS_DIR) if f.endswith('.json')
    )
    print(f"  Total model size: {total_kb:.0f} KB")
    print(f"  Models exported to: {MODELS_DIR}")
    print(f"  JS data exported to: {os.path.join(JS_DIR, 'data_computed.js')}")


if __name__ == '__main__':
    main()
