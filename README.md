## Diamond Price Prediction with Regime Modeling
---
Overview:

This project builds a machine learning pipeline to predict diamond prices using advanced feature engineering and probabilistic clustering.

---

The core idea is to:

- Model the baseline price–carat relationship
- Learn residual structure
- Discover hidden regimes using a Gaussian Mixture Model (GMM)

This creates a hybrid model that captures both:

- global trends
- local nonlinear patterns

---

### Feature Engineering

The pipeline constructs several categories of features:

1. Geometric Features
volume = x * y * z
geom_vol (std of dimensions)
Ratios:
```python
xy_ratio, xz_ratio, yz_ratio
```

2. Log Transformations
- log_carat
- log_volume

These stabilize variance and linearize relationships.

---


3. Polynomial Features
- carat_sq
- carat_cu

Capture nonlinear price scaling.

---

4. Interaction Features

- carat_depth
- carat_table


---

5. Local Density Features

Derived from sorted carat values:

- carat_diff
- local_density = 1 / (|Δcarat| + ε)

These approximate market density at different carat ranges.

---


6. Base Model (Trend Extraction)

A linear regression is trained:

- log_price ~ log_carat
---

Outputs:

- base_pred (expected log price)
- residual (deviation from trend)

7. Regime Detection (GMM)

A Gaussian Mixture Model clusters diamonds into hidden regimes using:

- size
- proportions
 -derived features


---

Outputs:
```python
regime (cluster label)
regime_prob_i (soft probabilities)
geom_regime_i (interaction with geometry)
```

This allows the model to learn market segmentation.

---

## Model Architecture

Final model:

- Gradient boosting model (e.g., CatBoost / similar)

Inputs:
- engineered features
- base predictions
- residuals
- regime probabilities

---

### Results: 
## Linear Models
```markdown
| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|------|-------------------|------------------|-----------------|
| Ridge | 1507.43 ± 454.43 | 814.94 ± 11.12 | 0.8417 ± 0.1115 |
| SVR   | 2832.45 ± 56.95  | 2022.65 ± 25.48 | 0.4935 ± 0.0152 |
```
## Tree-Based Models
```markdown
| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|------|-------------------|------------------|-----------------|
| Random Forest | 766.21 ± 13.85 | 388.99 ± 5.32 | 0.9629 ± 0.0006 |
| Gradient Boosting | 626.77 ± 10.10 | 341.47 ± 4.77 | 0.9752 ± 0.0009 |
```

## Boosted Models (Best Performance)
```markdown
| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|------|-------------------|------------------|-----------------|
| LightGBM | 554.08 ± 10.78 | 273.48 ± 3.25 | 0.9806 ± 0.0005 |
| XGBoost  | 538.27 ± 4.29  | 270.42 ± 3.49 | 0.9817 ± 0.0003 |
| CatBoost | 531.96 ± 5.47  | 274.09 ± 2.43 | 0.9821 ± 0.0004 |
```

## Key Findings:
- Boosted tree models dominate performance
- CatBoost achieved the best overall RMSE
- Extremely low variance across folds → strong generalization
- Feature engineering + regime modeling significantly improved results

---

## Key Insights:
- Log transformations significantly improved stability
- Residual modeling captured nonlinear pricing behavior
- GMM regimes helped separate pricing structures across diamond types
- Combining global + local structure improved prediction accuracy


---

### Important Design Decisions
- No Row Dropping

- All rows are preserved to maintain alignment with submission IDs.

Invalid values are handled via:

```python
replace([inf, -inf], nan).fillna(0)
```

- Index Preservation

- Sorting is used for feature computation but original row order is restored to prevent misalignment.


---

## Usage
```python
df, state = diamond_features(train_df, fit=True)

test_df, _ = diamond_features(test_df, fit=False, state=state)

preds = model.predict(test_df)
```

---

## Future Improvements
- Cross-validation with grouped splits
- Better density estimation (kernel density instead of diff)
- Regime-specific models (mixture of experts)
- Causal structure modeling

