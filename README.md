# Churn Model Portfolio Project

This repository now includes an end-to-end churn modeling workflow designed to be easy to explain in interviews.

## What this project demonstrates

- Business framing of churn prediction from exploratory analysis to model-ready data.
- Reproducible model training with feature preprocessing in a single scikit-learn pipeline.
- Model selection using cross-validation and validation metrics.
- Cost-aware threshold tuning (to connect model decisions to business trade-offs).
- Basic subgroup sanity checks (`gender`, `SeniorCitizen`) to discuss responsible AI considerations.
- Production-style inference options (API + batch CLI).

## Project structure

- `notebooks/` - EDA, cleaning, bivariate analysis, feature engineering exploration.
- `src/train_churn_model.py` - commented training pipeline + report/artifact generation.
- `src/inference_api.py` - FastAPI prediction service.
- `src/predict_cli.py` - batch prediction from CSV.
- `models/` - saved model artifacts and metrics (generated after training).
- `presentation/` - Claude-ready presentation brief and model evaluation report.

## Quick start

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Train model and generate artifacts:

```powershell
python src/train_churn_model.py
```

3. Regenerate Claude presentation package:

```powershell
python src/build_claude_presentation_pack.py
```

## Run the API

```powershell
uvicorn src.inference_api:app --reload
```

Test endpoints:

- `GET /health`
- `POST /predict`

Example payload for `/predict`:

```json
{
  "records": [
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "DSL",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 56.95,
      "TotalCharges": 617.35
    }
  ],
  "return_probabilities": true
}
```

## Run batch predictions

```powershell
python src/predict_cli.py --input-csv data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-csv models/predictions.csv
```

## Interview talking points

- Why threshold optimization matters more than a fixed 0.50 cutoff in churn use-cases.
- How class imbalance affects precision/recall trade-offs.
- Why preprocessing is embedded in the pipeline (training-serving consistency).
- How you would extend this with model registry, drift monitoring, and retraining automation.
