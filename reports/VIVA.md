# Viva preparation

**1. Why not just report accuracy?** 82.1% of test accidents are Slight; predicting Slight always scores 82.1% and misses every fatality. Macro-F1, recall, ROC-AUC, PR-AUC are imbalance-aware.
**2. Why a time-based split?** Deployment predicts the future. Random splits mix years, leak local patterns and inflate scores.
**3. How did you prevent leakage?** Dropped casualty count and police attendance (consequences). Spatial history uses only earlier years (expanding window); 2005 only seeds history. The ablation quantifies the leakage columns.
**4. Why did resampling not help?** After tuning per-class decision scales on 2015, all strategies reach similar macro-F1; threshold tuning is cheaper and cleaner than SMOTE.
**5. Why XGBoost/LightGBM?** Best on tabular mixed data, fast, handle sparse one-hot, exact TreeSHAP. They tie closely; XGBoost won on validation macro-F1.
**6. Why is performance modest?** Scene-only features carry limited signal (ROC-AUC ~0.70); vehicle, speed, driver and casualty detail are absent. Distribution shift in 2016-17 also hurts.
**7. What is calibration and why?** Class weighting distorts probabilities; isotonic calibration on 2015 cuts ECE from 0.137 to 0.025 so probabilities mean something.
**8. Why not DBSCAN for black spots?** On road networks it chains along corridors (huge clusters). A fixed 150 m severity-weighted peak detector gives compact, interpretable sites; DBSCAN/KDE are cross-checks.
**9. How are black spots validated?** Detect on 2005-14, check persistence on 2015-17 (95% top-100), rolling check, control sites, sensitivity to radius and weights. Persistence is lenient, so rank correlation is reported too.
**10. SHAP vs LIME?** SHAP is exact for trees and deterministic; LIME is a random local surrogate whose top features vary with seed (stability reported).
**11. What does SHAP not tell you?** Association, not causation; no interventions can be inferred without before-after studies.
**12. Limitations/ethics?** UK-only, injury-only, no traffic exposure, reporting change in 2016, possible reporting bias; predictions inform planning, never blame.
**13. How would you apply it to India?** Retrain on MoRTH/state data; the pipeline, black-spot method and app transfer, the model does not.
**14. What did you contribute?** End-to-end leak-free framework, imbalance study, validated black spots, SHAP/LIME comparison, in-browser exact TreeSHAP with parity test.
