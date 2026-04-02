// ─── Reference Data ──────────────────────────────────────────────────────────

const DISEASE_CLASSES = [
  {
    id: "t2dm",
    label: "Type 2 Diabetes",
    color: "#f59e0b",
    icon: "🩸",
    description: "Characterized by insulin resistance and elevated fasting glucose levels.",
    risk_factors: ["High BMI", "Elevated Glucose", "Sedentary Lifestyle", "Family History"]
  },
  {
    id: "hypertension",
    label: "Hypertension",
    color: "#ef4444",
    icon: "💓",
    description: "Persistently elevated arterial blood pressure requiring pharmacological intervention.",
    risk_factors: ["High Systolic BP", "High Diastolic BP", "Age", "High Sodium Diet"]
  },
  {
    id: "cvd",
    label: "Cardiovascular Disease",
    color: "#dc2626",
    icon: "🫀",
    description: "Structural or functional cardiac abnormalities associated with lipid dysregulation.",
    risk_factors: ["High LDL Cholesterol", "Low HDL Cholesterol", "Smoking", "Hypertension"]
  },
  {
    id: "stroke",
    label: "Stroke",
    color: "#7c3aed",
    icon: "🧠",
    description: "Cerebrovascular event caused by interrupted blood flow to the brain, strongly associated with hypertension and atrial fibrillation.",
    risk_factors: ["Hypertension", "Heart Disease", "Age", "Smoking", "High BMI"]
  },
  {
    id: "metabolic",
    label: "Metabolic Syndrome",
    color: "#8b5cf6",
    icon: "⚡",
    description: "Cluster of conditions including central obesity, dyslipidemia, and insulin resistance.",
    risk_factors: ["High BMI", "Elevated Triglycerides", "Low HDL", "Elevated Glucose"]
  },
  {
    id: "obesity",
    label: "Obesity-related Disorder",
    color: "#f97316",
    icon: "⚖️",
    description: "Excess adipose tissue accumulation increasing disease risk across multiple systems.",
    risk_factors: ["BMI > 30", "Low Physical Activity", "High Caloric Intake", "Hormonal Imbalance"]
  },
  {
    id: "healthy",
    label: "Healthy (Control)",
    color: "#10b981",
    icon: "✅",
    description: "Parameters within normal clinical ranges; suitable as control group for Phase 1 trials.",
    risk_factors: []
  }
];

const DIET_PLANS = {
  mediterranean: {
    id: "mediterranean",
    label: "Mediterranean Diet",
    icon: "🫒",
    color: "#06b6d4",
    description: "Rich in healthy fats, seafood, vegetables, legumes, and whole grains. Reduces cardiovascular risk and inflammation.",
    components: ["Olive oil", "Fish & seafood", "Whole grains", "Fresh vegetables", "Legumes", "Low-fat dairy"],
    avoid: ["Processed meats", "Refined sugars", "Trans fats"],
    calories: "1800–2200 kcal/day",
    studies: "Reduces CVD mortality by 30% (PREDIMED Study)"
  },
  dash: {
    id: "dash",
    label: "DASH Diet",
    icon: "🥗",
    color: "#10b981",
    description: "Dietary Approaches to Stop Hypertension. Emphasizes low sodium, high potassium, and lean proteins.",
    components: ["Low-sodium foods", "Potassium-rich vegetables", "Lean proteins", "Low-fat dairy", "Whole grains"],
    avoid: ["High-sodium foods", "Saturated fats", "Alcohol"],
    calories: "2000 kcal/day",
    studies: "Reduces SBP by 11 mmHg in hypertensive patients"
  },
  low_glycemic: {
    id: "low_glycemic",
    label: "Low-Glycemic Diet",
    icon: "📉",
    color: "#f59e0b",
    description: "Focuses on slow-digesting carbohydrates to maintain stable blood glucose levels and insulin sensitivity.",
    components: ["Legumes", "Non-starchy vegetables", "Berries", "Oats", "Sweet potatoes", "Nuts"],
    avoid: ["White bread", "Sugary drinks", "White rice", "Processed snacks"],
    calories: "1600–1900 kcal/day",
    studies: "Reduces HbA1c by 0.5% in T2DM patients"
  },
  heart_healthy: {
    id: "heart_healthy",
    label: "Heart-Healthy Diet",
    icon: "❤️",
    color: "#ef4444",
    description: "Optimized to lower LDL cholesterol and triglycerides while increasing protective HDL levels.",
    components: ["Fatty fish (omega-3)", "Flaxseeds", "Walnuts", "Plant sterols", "Fiber-rich foods"],
    avoid: ["Trans fats", "High cholesterol foods", "Excessive saturated fats"],
    calories: "1800–2000 kcal/day",
    studies: "Reduces LDL by 15–20% without medication"
  },
  ketogenic: {
    id: "ketogenic",
    label: "Ketogenic Diet",
    icon: "🥑",
    color: "#8b5cf6",
    description: "Very low-carbohydrate, high-fat diet inducing ketosis for rapid weight loss and metabolic benefits.",
    components: ["Avocado", "Eggs", "Cheese", "Meat & fish", "Low-carb vegetables", "Nuts & seeds"],
    avoid: ["Bread & pasta", "Sugar", "Fruit", "Starchy vegetables", "Grains"],
    calories: "1500–1800 kcal/day",
    studies: "Reduces BMI by 3–5 units in 12 weeks"
  },
  balanced: {
    id: "balanced",
    label: "Balanced Caloric Diet",
    icon: "⚖️",
    color: "#64748b",
    description: "Standard balanced macronutrient distribution suitable for healthy individuals with no specific restrictions.",
    components: ["Mixed whole grains", "Lean proteins", "Colorful vegetables", "Moderate healthy fats", "Fruits"],
    avoid: ["Excess processed foods", "Empty calories"],
    calories: "2000–2500 kcal/day",
    studies: "Maintains metabolic homeostasis in healthy controls"
  }
};

const DRUG_CLASSES = {
  ace_inhibitors: {
    id: "ace_inhibitors",
    label: "ACE Inhibitors",
    icon: "💊",
    color: "#ef4444",
    mechanism: "Inhibit angiotensin-converting enzyme, reducing blood pressure by vasodilation.",
    examples: ["Lisinopril", "Enalapril", "Ramipril", "Captopril"],
    indication: "Hypertension, Heart Failure, Diabetic Nephropathy",
    contraindications: ["Pregnancy", "Bilateral renal artery stenosis", "Hyperkalemia"],
    phase1_focus: "Dose-finding for renal protective effects"
  },
  statins: {
    id: "statins",
    label: "HMG-CoA Statins",
    icon: "🧬",
    color: "#dc2626",
    mechanism: "Inhibit HMG-CoA reductase, reducing hepatic cholesterol synthesis.",
    examples: ["Atorvastatin", "Rosuvastatin", "Simvastatin", "Pravastatin"],
    indication: "Hypercholesterolemia, Cardiovascular Disease Prevention",
    contraindications: ["Active liver disease", "Pregnancy", "Myopathy history"],
    phase1_focus: "Bioavailability and hepatic metabolism profiling"
  },
  metformin_class: {
    id: "metformin_class",
    label: "Biguanides (Metformin-class)",
    icon: "⚗️",
    color: "#f59e0b",
    mechanism: "Reduces hepatic glucose production and improves peripheral insulin sensitivity.",
    examples: ["Metformin", "Phenformin (research)", "Novel biguanide analogs"],
    indication: "Type 2 Diabetes Mellitus, Pre-diabetes, Metabolic Syndrome",
    contraindications: ["Renal impairment (eGFR < 30)", "Lactic acidosis risk"],
    phase1_focus: "Glucose lowering dose-response in insulin-resistant patients"
  },
  beta_blockers: {
    id: "beta_blockers",
    label: "Beta-Blockers",
    icon: "🛡️",
    color: "#8b5cf6",
    mechanism: "Block β-adrenergic receptors, reducing heart rate and myocardial contractility.",
    examples: ["Metoprolol", "Carvedilol", "Bisoprolol", "Atenolol"],
    indication: "Hypertension, Angina, Heart Failure, Arrhythmia",
    contraindications: ["Severe bradycardia", "Asthma", "2nd/3rd degree AV block"],
    phase1_focus: "Cardiac safety and HR modulation in healthy volunteers"
  },
  glp1_agonists: {
    id: "glp1_agonists",
    label: "GLP-1 Receptor Agonists",
    icon: "🔬",
    color: "#10b981",
    mechanism: "Mimic incretin hormone GLP-1, enhancing glucose-dependent insulin secretion.",
    examples: ["Semaglutide", "Liraglutide", "Dulaglutide", "Exenatide"],
    indication: "Type 2 Diabetes, Obesity, Cardiovascular Risk Reduction",
    contraindications: ["Medullary thyroid carcinoma", "MEN2", "Pancreatitis history"],
    phase1_focus: "Weight loss and glycemic control in obese T2DM patients"
  },
  no_intervention: {
    id: "no_intervention",
    label: "Lifestyle Intervention Only",
    icon: "🌿",
    color: "#06b6d4",
    mechanism: "Structured exercise, dietary modification, and behavioral change without pharmacotherapy.",
    examples: ["Exercise prescription", "Dietary counseling", "Behavior modification programs"],
    indication: "Healthy controls, early-stage metabolic risk",
    contraindications: [],
    phase1_focus: "Placebo-controlled baseline assessment for trial integrity"
  }
};

const FEATURE_IMPORTANCE = [
  { feature: "Blood Glucose Level", importance: 0.187, category: "Physiological" },
  { feature: "BMI", importance: 0.163, category: "Physiological" },
  { feature: "LDL Cholesterol", importance: 0.142, category: "Physiological" },
  { feature: "Systolic Blood Pressure", importance: 0.128, category: "Physiological" },
  { feature: "Age", importance: 0.098, category: "Demographic" },
  { feature: "HDL Cholesterol", importance: 0.087, category: "Physiological" },
  { feature: "Physical Activity Level", importance: 0.072, category: "Lifestyle" },
  { feature: "Diastolic Blood Pressure", importance: 0.054, category: "Physiological" },
  { feature: "Weekly Exercise Hours", importance: 0.038, category: "Lifestyle" },
  { feature: "Family History", importance: 0.031, category: "Medical History" },
  { feature: "Gender", importance: 0.024, category: "Demographic" },
  { feature: "Smoking Status", importance: 0.018, category: "Lifestyle" },
  { feature: "Heart Rate", importance: 0.014, category: "Physiological" },
  { feature: "Alcohol Consumption", importance: 0.008, category: "Lifestyle" }
];

const POPULATION_NORMS = {
  bmi: { mean: 27.5, sd: 5.2 },
  glucose: { mean: 99.0, sd: 18.5 },
  ldl: { mean: 130, sd: 35 },
  hdl: { mean: 52, sd: 14 },
  systolic: { mean: 124, sd: 18 },
  diastolic: { mean: 80, sd: 12 },
  heart_rate: { mean: 72, sd: 11 },
  exercise_hours: { mean: 3.5, sd: 2.8 }
};
