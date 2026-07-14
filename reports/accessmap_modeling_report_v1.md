# AccessMap: Modeling & Retrieval — Status Report

## What's Done
Notebook 3 pipeline runs end to end. Both PMR and Housing datasets have models, embeddings, and search indexes. 9 files saved and verified.

## PMR Model (Amsterdam Sidewalks)
- Accuracy: 84% (baseline: 81%)
- Problem: Model over-flags segments as "not accessible." Recall is perfect (1.00), precision is weak (0.55).
- Label note: We built our own accessibility label from width + curb height. Real labels only exist for 50 rows (too few to train on).

## Housing Model (US Addresses)
- Accuracy: 67% (baseline: 85%). Worse than just guessing.
- Root cause: `state` accounts for 89% of feature importance. Model is mostly learning geography, not house condition.
- This is a bias problem, not a tuning problem.

## FAISS Search
- Housing: works well. Similar addresses return meaningful, ranked results.
- PMR: groups similar segments correctly, but many exact ties. Caused by low-variety features (many sidewalks share the same width/crossing/curb values). Not fixable without adding location data.

## Decision Needed
Before fine-tuning, we need to decide how to fix the Housing bias:
1. Drop `state` and retrain — see how much accuracy drops.
2. Keep `state` but treat the model as "mostly predicting by region," not by house condition.
3. Try a different model type (not Random Forest) for Housing.

## Next Step
Fix Housing bias first. Then fine-tune both models (hyperparameters, thresholds).
