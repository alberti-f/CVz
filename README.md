# cleanfit

A lightweight Python utility for **confound-controlled modelling**. Built for personal use in neuroscience and psychology research, where the core question is:

> *Does X predict y, after accounting for nuisance variables (C)?*

The package handles covariate residualization, scaling, cross-validated model evaluation, permutation testing, and hyperparameter search — all in a leakage-free, train/test-honest way.

---

## Functions

| Function | Description |
|---|---|
| `run_cv` | Cross-validated model fitting and evaluation with optional residualization |
| `residualize_splits` | Leakage-free residualization of train/test splits |
| `scale_splits` | Fit-transform train, transform test for any number of array pairs |
| `permutation_test` | Permutation inference on median cross-validated performance |
| `grid_search` | Hyperparameter search over a parameter grid |
| `summarize_results` | Summarize metrics across CV folds |

---

## Usage

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from cleanfit import run_cv, permutation_test, summarize_results

X = np.random.randn(100, 50)   # features
y = np.random.randn(100)       # target
C = np.random.randn(100, 3)    # covariates (e.g. age, sex, site)

model = Ridge(alpha=1.0)
split = KFold(n_splits=5, shuffle=True, random_state=0)

# Cross-validated prediction after residualizing y (and optionally X) on C
results = run_cv(X, y, model, split, C=C, residualize_X=False)
summary = summarize_results(results)

# Permutation test
perm = permutation_test(X, y, model, split, C=C, n_perm=1000, metric="r")
print(f"r = {perm['real']:.3f}, p = {perm['p']:.3f}")
```

---

## Dependencies

- `numpy`
- `scikit-learn`
- `joblib`

Install with:

```bash
pip install -r requirements.txt
```

---

## Notes

- Developed for personal research use. No guarantees of stability or API consistency across versions.
- Residualization uses OLS (`np.linalg.lstsq`) with an intercept by default.
- Covariates are scaled before residualization; features are scaled after.
- `summarize_results` computes per-fold median and std for all numeric outputs returned by `_eval_dict`, including model coefficients where available.