import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.base import clone
from joblib import Parallel, delayed

def _scale_pair(tr, te):
    s = StandardScaler()
    return s.fit_transform(tr), s.transform(te)

def scale_splits(*pairs):
    """
    Fit-transform train, transform test for any number of array pairs.

    Parameters
    ----------
    *pairs : (train_array, test_array)
        Each pair is independently scaled.


    Returns
    -------
    List of (scaled_train, scaled_test) tuples, one per input pair.

    Example
    -------
    (Xtr, Xte), (Ctr, Cte), (ytr, yte) = scale_splits(
        (Xtr, Xte), (Ctr, Cte), (ytr[:, None], yte[:, None])
    )
    """
    return [_scale_pair(tr, te) for tr, te in pairs]


def _eval_dict(fitted_model, Xte, yte):
    """
    Extract predictions, metrics, and all numeric learned attributes
    from a fitted sklearn-like estimator.

    Returns
    -------
    dict
        Flexible container with metrics, predictions, and model attributes.
    """

    m = getattr(fitted_model, "model_", fitted_model)

    y_pred = m.predict(Xte)
    y_true = np.atleast_1d(yte)
    y_pred = np.asarray(y_pred)

    out = {
        "y_true": y_true,
        "y_pred": y_pred,
        "r": np.corrcoef(y_true, y_pred)[0, 1],
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
    }

    for attr in dir(m):
        if not attr.endswith("_") or attr.startswith("__"):
            continue

        val = getattr(m, attr)
        if isinstance(val, np.ndarray):
            out[attr] = val

        elif isinstance(val, (int, float, np.floating, np.integer)):
            out[attr[:-1]] = val

    return out


def _add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def residualize_splits(Ytr, Yte, Xtr, Xte, fit_intercept=True):
    """
    Residualize train and test data using coefficients estimated on train only.

    Parameters
    ----------
    tr, te : ndarray
        Train and test data to residualize. Can be 1D or 2D.
    Ctr, Cte : ndarray
        Train and test covariates.
    fit_intercept : bool, default True
        Whether to include an intercept in the covariate model.

    Returns
    -------
    tr_res, te_res : ndarray
        Residualized train and test data with original dimensionality preserved.
    """
    Ytr = np.asarray(Ytr)
    Yte = np.asarray(Yte)
    Ytr_2d = Ytr[:, None] if Ytr.ndim == 1 else Ytr
    Yte_2d = Yte[:, None] if Yte.ndim == 1 else Yte

    if fit_intercept:
        Xtr, Xte = _add_intercept(Xtr), _add_intercept(Xte)

    beta = np.linalg.lstsq(Xtr, Ytr_2d, rcond=None)[0]
    Ytr_res = Ytr_2d - Xtr @ beta
    Yte_res = Yte_2d - Xte @ beta

    return np.atleast_1d(Ytr_res.squeeze()), np.atleast_1d(Yte_res.squeeze())


def run_cv(X, y, model, split, C=None, residualize_X=False):
    """
    Cross-validated analysis with optional covariate residualization and scaling.

    Parameters

    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector.
    model : estimator 
        sklearn-like estimator with fit and predict methods. 
    splits : int, default 5
        Number of CV folds.
    C : ndarray, optional
        Covariate matrix for residualization. If None, no residualization is performed.
    seed : int, default 0
        Random seed for reproducibility.
    residualize_X : bool, default False
        Whether to residualize features in addition to the target. If True, X
        is residualized using the same covariates C and train/test split as y.
    """

    res = []

    for tr, te in split.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        if C is not None:
            Ctr, Cte = C[tr], C[te]
            Ctr, Cte = scale_splits((Ctr, Cte))[0]

            ytr, yte = residualize_splits(ytr, yte, Ctr, Cte, fit_intercept=True)
            if residualize_X:
                Xtr, Xte = residualize_splits(Xtr, Xte, Ctr, Cte, fit_intercept=True)
        Xtr, Xte = scale_splits((Xtr, Xte))[0]

        m = clone(model)
        if hasattr(m, "fit_intercept"):
            m.set_params(fit_intercept=True) # Force no intercept, data is centered (but double-check)
        m.fit(Xtr, ytr)
        res.append(_eval_dict(m, Xte, yte))

    return res


def summarize_results(res):
    """
    Summarize CV results from run_cv (list of dicts).
    """

    summary = {}
    keys = set().union(*[d.keys() for d in res])
    keys -= {"y_true", "y_pred"}

    for k in keys:
        vals = [d[k] for d in res if k in d]

        if not vals:
            continue

        try:
            arr = np.stack(vals)
            summary[f"median_{k}"] = np.median(arr, axis=0)
            summary[f"std_{k}"] = arr.std(axis=0)
        except Exception:
            continue

    return summary


def permutation_test(X, y, model, split, C=None, n_perm=1000, seed=0, metric="r", residualize_X=False):
    rng = np.random.default_rng(seed)

    def score(y_):
        res = run_cv(X, y_, model, split, C=C, residualize_X=residualize_X)
        return np.median([d[metric] for d in res])

    real = score(y)

    null = np.array(
        Parallel(n_jobs=-1)(
            delayed(score)(rng.permutation(y)) for _ in range(n_perm)
        )
    )

    operator = np.greater_equal if metric in ["r", "r2"] else np.less_equal
    p = operator(null, real).mean()

    return {
        "real": real,
        "null": null,
        "p": p
    }


def grid_search(param_grid, X, y, model, split, C=None, metric="r"):

    from itertools import product

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    results = []

    for combo in combos:
        params = dict(zip(keys, combo))

        m = clone(model).set_params(**params)

        res = run_cv(X, y, m, split, C=C, residualize_X=False)
        score = np.median([d[metric] for d in res])

        results.append({
            "params": params,
            "score": score
        })

    return results