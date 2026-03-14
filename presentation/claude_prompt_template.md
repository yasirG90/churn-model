# Claude Prompt Template (Copy/Paste)

You are a senior AI/ML communicator preparing a high-quality presentation for hiring-panel review.

Create a **12-slide deck** from the provided project brief.

## Audience

- Primary: Lead AI Engineer interview panel
- Secondary: business stakeholders interested in churn reduction outcomes

## Tone and style

- Executive-friendly, concise, and evidence-based
- Confident but not exaggerated
- Explain trade-offs clearly (precision vs recall, threshold choice, risk controls)

## Required output for each slide

For each of the 12 slides provide:

1. Slide title
2. 3-5 bullets
3. Suggested visual/chart
4. Speaker notes (4-6 sentences, natural and interview-ready)

## Non-negotiable content coverage

Include all of the following:

1. Business objective and why churn prediction matters
2. Data source, cleaning logic, and class imbalance
3. EDA and bivariate insights that motivated modeling choices
4. Modeling pipeline architecture (`ColumnTransformer`, CV, model comparison)
5. Candidate model comparison with exact metrics
6. Final selected model and rationale
7. Threshold tuning and cost assumptions (FN=5, FP=1)
8. Subgroup sanity checks and responsible AI considerations
9. Inference/deployment assets (API + CLI + artifacts)
10. Business impact quantification framework (campaign capacity, offer success assumptions, net impact formula)
11. Governance and operating model (owners, monitoring cadence, retraining triggers, change control)
12. Risks, limitations, and concrete roadmap

## Accuracy rules

- Use exact metrics from the brief/report; do not invent numbers.
- If any detail is missing, explicitly state the assumption.
- Keep technical claims grounded in project evidence.

## Final slide requirement

End with a "Why this is lead-level" slide summarizing:

- technical depth
- decision quality
- production readiness
- governance awareness

Now use the project brief below as the source of truth.