# Early Warning Machine Learning Models for Sepsis Prediction in Intensive Care Units

> M.Sc. Data Science Thesis — University of Europe for Applied Sciences, Potsdam (2026)  
> Author: Agrajeet Shashivikas Vaze

---

## Overview

Sepsis kills approximately 11 million people per year — 20% of all global deaths. Traditional clinical scoring systems (SOFA, qSOFA, SIRS) flag patients too late and fail to capture the physiological complexity and heterogeneity of the condition.

This project builds and evaluates interpretable machine learning models that can predict sepsis **6, 12, and 24 hours before clinical diagnosis** using routine ICU data — vital signs and laboratory values already available in electronic health records.

The key finding: a logistic regression model achieves **AUROC 0.876** at the 6-hour prediction horizon, significantly outperforming a Random Forest (AUROC 0.684) under the same rigorous patient-level evaluation protocol.

---

## Results

| Prediction Horizon | AUROC | AUPRC | 95% CI (AUROC) |
|---|---|---|---|
| **6 hours** | **0.876** | 0.062 | (0.869, 0.883) |
| 12 hours | 0.854 | 0.082 | (0.846, 0.862) |
| 24 hours | 0.850 | 0.161 | (0.842, 0.858) |

**Logistic Regression vs Random Forest (12-hour horizon)**

| Model | AUROC | AUPRC | Precision @ 90% Sensitivity | Recall @ 90% Precision |
|---|---|---|---|---|
| Logistic Regression | 0.854 | 0.082 | 0.18 | 0.67 |
| Random Forest | 0.684 | 0.025 | 0.09 | 0.31 |

The simpler, interpretable logistic regression outperforms the more complex ensemble model — a counterintuitive but clinically significant finding explained by strict patient-level data splitting, class imbalance, and the model's better generalisation on unseen patients.

---

## Dataset

**PhysioNet Computing in Cardiology Challenge 2019**  
- 40,000+ ICU patient electronic health records  
- 40 clinical features: vital signs (heart rate, blood pressure, temperature, SpO2, respiratory rate) + laboratory values (lactate, bilirubin, creatinine, FiO2, magnesium, haemoglobin, pH, etc.)  
- Sepsis labelled using Sepsis-3 criteria  
- ~7% sepsis prevalence (severe class imbalance)

Dataset: [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)

---

## Methodology

### Problem Formulation
Three separate binary classification tasks — predict sepsis onset within 6, 12, or 24 hours of the current timestep, using only data available up to that point (no lookahead).

### Preprocessing Pipeline
- **Missing data**: forward-fill within patient stays + median imputation for remaining gaps
- **Feature engineering**: time-sensitive features appropriate to each prediction horizon
- **Exclusion criteria**: patients with sepsis already present at ICU admission removed
- **Data split**: patient-level (no patient appears in both train and test sets — prevents information leakage)

### Models
- **Logistic Regression** (primary): L2 regularisation, C tuned via 5-fold cross-validation grid search (log scale 0.001–100). Chosen for interpretability and clinical deployability.
- **Random Forest** (comparison): randomised hyperparameter search (n_estimators: 100/300/500, max_depth: 10/20/30/None, min_samples_leaf: 1/5/10), 50 random configurations.

### Evaluation
- Patient-level train/test split to prevent information leakage
- Bootstrap confidence intervals (resampled test set, 1000 iterations)
- Metrics: AUROC, AUPRC, Precision/Recall at clinically meaningful operating points
- AUPRC used alongside AUROC to account for severe class imbalance

### Explainability Analysis
Logistic regression coefficients analysed across all three horizons:
- **pH, FiO₂, magnesium, lactate** — consistently strong predictors across all horizons
- **Bilirubin, temperature, haemoglobin** — informative at specific time windows before onset
- All top features align with known sepsis pathophysiology, confirming clinical validity

---

## Tech Stack

| Tool | Version |
|---|---|
| Python | 3.8 |
| scikit-learn | 1.7.0 |
| NumPy | 2.3.1 |
| pandas | 2.3.1 |
| Matplotlib | 3.10.3 |
| Jupyter Notebook | — |

All experiments run on CPU (Intel x86-64, 16 GB RAM) — no GPU required. Logistic regression training: seconds per horizon. Random Forest: minutes per horizon.

---

## Key Findings

1. **Sepsis is detectable up to 24 hours before clinical diagnosis** using only routine ICU measurements
2. **Interpretable models can outperform complex ones** when patient-level evaluation is strictly applied — a finding with direct implications for clinical AI deployment
3. **Physiologically meaningful features dominate predictions**: the model learned real clinical signals, not statistical artefacts
4. **The model is clinically flexible**: institutions can adjust the operating threshold to prioritise sensitivity (catch more cases) or precision (reduce alert fatigue) based on their specific context

---

## Clinical Significance

A 6-hour early warning window gives ICU teams meaningful lead time for intervention — fluid resuscitation, antibiotic escalation, and closer monitoring. Unlike black-box deep learning models, logistic regression coefficients can be directly inspected and explained to clinicians, which is a prerequisite for real-world adoption in regulated healthcare environments.

---

## Repository Structure

```
├── notebooks/
│   ├── 01_preprocessing.ipynb      # Data cleaning, imputation, feature engineering
│   ├── 02_model_training.ipynb     # LR and RF training + hyperparameter tuning
│   ├── 03_evaluation.ipynb         # AUROC, AUPRC, bootstrap CIs, operating points
│   └── 04_explainability.ipynb     # Coefficient analysis, feature ranking by horizon
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── results/
│   └── figures/                    # AUROC curves, precision-recall curves, feature importance plots
└── README.md
```

> **Note**: Raw PhysioNet data cannot be redistributed. Download from [physionet.org](https://physionet.org/content/challenge-2019/1.0.0/) and place in `data/raw/` to reproduce.

---

## Citation

```
Vaze, A.S. (2026). Early Warning Machine Learning Models for Sepsis Prediction 
in Intensive Care Units. M.Sc. Thesis, University of Europe for Applied Sciences, 
Potsdam. Supervisors: Prof. Dr. Iftikhar Ahmed, Prof. Dr. Talha Ali.
```

---

## About

This project is part of an M.Sc. in Data Science at the University of Europe for Applied Sciences. It sits at the intersection of clinical data science, healthcare AI, and patient safety — areas I work in professionally as a Digital Health Analyst.

Connect: [linkedin.com/in/agrajeet-vaze](https://linkedin.com/in/agrajeet-vaze)
