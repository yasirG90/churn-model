# Churn Model Evaluation Report

Generated from current project artifacts and validation outputs.

## 1) Model Selection Summary

Two candidate models were evaluated using 5-fold stratified cross-validation on the training split and then validated on a held-out set.

| Model | CV ROC-AUC (mean) | Validation ROC-AUC | Validation Precision | Validation Recall | Validation F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8459 | 0.8351 | 0.4901 | 0.7968 | 0.6069 |
| Random Forest | 0.8415 | 0.8255 | 0.5618 | 0.6444 | 0.6002 |

**Selected model:** Logistic Regression

Rationale:

- Higher validation ROC-AUC than Random Forest
- Better recall profile for churn-prevention use case
- More interpretable baseline for business communication

## 2) Thresholding Strategy

Default threshold `0.50` is not always cost-optimal in churn scenarios.

Assumed business costs used for threshold tuning:

- False Negative (missed churner): **5.0**
- False Positive (unnecessary retention action): **1.0**

Cost-optimized threshold found on validation set: **0.33**

At threshold 0.33:

- Precision: **0.4245**
- Recall: **0.9171**
- F1: **0.5804**
- Expected misclassification cost: **620.00**

Interpretation: the tuned threshold intentionally prioritizes capturing more at-risk customers (high recall), accepting lower precision as a business trade-off.

## 3) Subgroup Sanity Check (Early Responsible AI Signal)

These checks are diagnostic only and do not replace a formal fairness assessment.

### Gender

- Female: count=681, positive_rate=0.590, precision=0.420, recall=0.904
- Male: count=726, positive_rate=0.559, precision=0.429, recall=0.930

### Senior Citizen

- 0 (non-senior): count=1175, positive_rate=0.522, precision=0.406, recall=0.896
- 1 (senior): count=232, positive_rate=0.841, precision=0.482, recall=0.979

## 4) What This Means for Deployment

- Use model probability as ranking signal for retention campaigns.
- Keep threshold configurable by budget and campaign capacity.
- Track subgroup metrics continuously to detect widening performance gaps.

## 5) Next Technical Upgrades

1. Add calibration analysis and decile lift charts.
2. Add temporal validation to test robustness over time.
3. Add post-deployment drift and performance monitoring.
4. Add ROI simulation by intervention budget.