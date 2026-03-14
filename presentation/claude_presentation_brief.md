# Churn Prediction Project Brief (Claude-Ready)

## 1) Executive Summary

This project predicts telecom customer churn and translates model output into decision-ready actions.

- **Business objective:** identify at-risk customers early enough to trigger retention interventions.
- **Dataset:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Raw rows:** 7,043
- **Rows used after cleaning:** 7,032
- **Target:** `Churn` (`Yes` / `No`)

The project now includes:

1. EDA and feature insights in notebooks (`01` to `05`)
2. Reproducible training pipeline in `src/train_churn_model.py`
3. Saved model + metrics artifacts in `models/`
4. Inference interfaces:
   - API (`src/inference_api.py`)
   - Batch CLI (`src/predict_cli.py`)

---

## 2) Problem Framing

### Why churn prediction matters

- Acquiring new customers is typically more expensive than retaining existing ones.
- Churn prediction allows prioritized retention campaigns rather than broad, costly outreach.
- The model is designed as a **decision support** tool (probability + threshold), not a replacement for business judgment.

### Success criteria

- Good ranking quality (ROC-AUC)
- Strong recall for churners (avoid missing high-risk customers)
- Explicit threshold strategy to align with retention costs

---

## 3) Data & Cleaning Decisions

### Source and structure

- Customer demographics, account tenure, services, billing, and contract attributes.
- Mixed numeric + categorical feature set.

### Key cleaning decisions

- `TotalCharges` converted to numeric (`errors='coerce'`)
- 11 rows with missing `TotalCharges` dropped
- `customerID` removed from modeling features (identifier leakage avoidance)

### Class balance

- Churn class is imbalanced (approximately 26.5% churn), which motivated:
  - class-weighted training
  - threshold tuning beyond 0.50

---

## 4) Exploratory Findings (Notebook Narrative)

### Univariate highlights

- Tenure shows concentration of newer customers and long-tenure customers.
- Monthly charges show multi-modal distribution.
- Churn distribution confirms imbalance.

### Bivariate highlights

- **Contract type:** month-to-month customers churn substantially more.
- **Internet service:** fiber customers show elevated churn risk in notebook analysis.
- **Demographics:** senior-citizen segment shows higher risk signals.

### Correlation / feature engineering direction

- Contract/billing/service combinations are likely high-signal interactions.
- Practical feature handling used in production pipeline:
  - numeric imputation + scaling
  - categorical imputation + one-hot encoding

---

## 5) Modeling Approach (Production Script)

Training is implemented in `src/train_churn_model.py` with a reproducible, interview-friendly pipeline.

### Pipeline design

- Train/validation split with stratification
- Preprocessing via `ColumnTransformer`
- 5-fold stratified cross-validation for model comparison
- Candidate models:
  - Logistic Regression (`class_weight='balanced'`)
  - Random Forest (`class_weight='balanced_subsample'`)

### Model selection (from `models/training_metrics.json`)

| Model | CV ROC-AUC (mean) | Validation ROC-AUC | Validation Precision | Validation Recall | Validation F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8459 | 0.8351 | 0.4901 | 0.7968 | 0.6069 |
| Random Forest | 0.8415 | 0.8255 | 0.5618 | 0.6444 | 0.6002 |

**Selected model:** Logistic Regression (best validation ROC-AUC and stronger recall profile for retention use-case).

---

## 6) Threshold Strategy & Cost Framing

### Why threshold tuning was added

Using a fixed 0.50 cutoff is often suboptimal in churn prevention.

Cost assumptions used in code:

- False Negative cost = 5.0 (missing a true churner is expensive)
- False Positive cost = 1.0 (unnecessary outreach is cheaper)

### Tuned threshold result

- **Optimized threshold:** 0.33
- **Expected misclassification cost:** 620.00

At threshold 0.33:

- Precision = 0.4245
- Recall = 0.9171
- F1 = 0.5804

Interpretation: this setup intentionally favors **high recall** to capture more likely churners for proactive retention.

---

## 7) Business Impact Quantification (CAIO Talking Points)

The project can be presented as a decision engine that optimizes retention action allocation.

### Example campaign simulation (framework for executive discussion)

Using validation predictions (`models/validation_predictions.csv`), estimate the impact at threshold 0.33 with:

- Campaign capacity scenarios (e.g., top 10%, 20%, 30% highest-risk customers)
- Expected saves under multiple offer-success assumptions (e.g., 10%, 20%, 30%)
- Budget assumptions per intervention (call center only vs incentive + discount)

### Suggested equations for presentation

- `Expected saves = Targeted true churners × Offer success rate`
- `Gross value saved = Expected saves × Average customer lifetime value`
- `Campaign cost = Targeted customers × Cost per intervention`
- `Net impact = Gross value saved - Campaign cost`

### Executive interpretation

- Threshold 0.33 is aligned to minimizing missed churners where false negatives are 5x costlier than false positives.
- The model should be pitched as a **rank-and-prioritize system** connected to campaign budget and capacity.

---

## 8) Responsible AI / Fairness Sanity Check

Basic subgroup checks were run on validation predictions (not a full fairness audit).

### Gender

- Female: precision 0.420, recall 0.904
- Male: precision 0.429, recall 0.930

### Senior Citizen

- Non-senior (`0`): precision 0.406, recall 0.896
- Senior (`1`): precision 0.482, recall 0.979

Discussion point: subgroup-level performance differences indicate where targeted review and policy guardrails should be strengthened before production rollout.

---

## 9) Artifacts and Deliverables

### Generated model assets

- `models/churn_model.joblib`
- `models/model_metadata.json`
- `models/training_metrics.json`
- `models/validation_predictions.csv`

### Inference assets

- `src/inference_api.py` (FastAPI: `/health`, `/predict`)
- `src/predict_cli.py` (batch CSV scoring)

### Reporting assets

- `presentation/model_evaluation_report.md`
- `presentation/claude_prompt_template.md`

---

## 10) Governance & Operating Model (CAIO Readiness)

### Proposed operating model

- **Model Owner:** Lead AI Engineer (technical health, retraining decisions)
- **Business Owner:** Retention/CX lead (campaign policy, budget, intervention strategy)
- **Risk/Governance:** periodic fairness/performance review with documented approvals

### Monitoring cadence

- Weekly: score distribution drift and campaign conversion metrics
- Monthly: precision/recall by segment, especially senior-citizen subgroup
- Quarterly: full model review and threshold recalibration

### Retraining triggers

- ROC-AUC drop greater than agreed tolerance (example: > 3-5% relative)
- Sustained precision or recall decline for 2+ monitoring windows
- Material upstream data distribution change in key drivers (contract, tenure, internet service)

### Change-control expectations

- Versioned model artifacts and metadata
- Signed-off threshold changes before production
- Rollback-ready deployment process

---

## 11) Risks, Limitations, and Next Steps

### Current limitations

- Single-dataset evaluation (no temporal validation)
- No calibration plot / lift chart in current report
- Fairness checks are basic and not policy-complete
- No online monitoring/drift pipeline yet

### Recommended next upgrades

1. Add calibration + decile lift analysis
2. Add drift monitoring plan (data drift + performance drift)
3. Add cost/ROI simulation by campaign budget
4. Define retraining trigger policy and model versioning workflow

---

## 12) Suggested Slide Sequence (12 Slides)

1. Business problem and objective
2. Dataset and target definition
3. Data quality and cleaning decisions
4. EDA highlights (univariate)
5. Bivariate drivers of churn
6. Feature strategy and preprocessing
7. Candidate model comparison
8. Final model performance
9. Threshold, cost strategy, and business impact simulation
10. Subgroup sanity findings + governance implications
11. Deployment-ready assets (API + CLI + artifacts)
12. Risks, operating model, and roadmap

---

## 13) Source of Truth for Claude

When generating slides, use these files as authoritative project evidence:

- `notebooks/01_Exploratory_Data_Analysis.ipynb`
- `notebooks/02_Data_Cleaning_and_Univariate_Analysis.ipynb`
- `notebooks/03_Bivariate_Analysis.ipynb`
- `notebooks/04_Correlation_and_Feature_Engineering.ipynb`
- `notebooks/05_Model_Building_and_Evaluation.ipynb`
- `models/training_metrics.json`
- `presentation/model_evaluation_report.md`

If anything is missing, Claude should state assumptions explicitly instead of inventing numbers.