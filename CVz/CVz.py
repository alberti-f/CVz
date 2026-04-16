import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.base import clone
from joblib import Parallel, delayed
from itertools import product


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
    (Xtr, Xte), (Ctr, Cte), (ytr[:, None], yte[:, None]) = scale_splits(
        (Xtr, Xte), (Ctr, Cte), (ytr[:, None], yte[:, None])
    )
    """
    return [_scale_pair(tr, te) for tr, te in pairs]


def _pairwise_corr(A, B, copy=True):
    """
    Compute column-wise correlation coefficients between two matrices.

    Parameters
    ----------
    A : numpy.ndarray
        First input matrix.
    B : numpy.ndarray
        Second input matrix.

    Returns
    -------
    numpy.ndarray
        Correlation coefficients for each column.
    """

    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of samples.")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns.")

    # center the matrices
    if copy:
        A = A.copy()
        B = B.copy()
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)

    # Calculate correlation coefficients
    numer = np.sum(A * B, axis=0)
    denom = np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)
    return np.divide(numer, denom, out=np.full(numer.shape, np.nan, dtype=float), where=denom > 0)


def _as_2d_columns(y):
    y = np.asarray(y)
    if y.ndim == 1:
        return y[:, None]
    return y


def prediction_metrics(y_true, y_pred):
    """
    Compute common regression metrics for true vs predicted values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    dict
        Dictionary containing correlation, R^2, MAE, and RMSE.
    """
    y_true = _as_2d_columns(y_true)
    y_pred = _as_2d_columns(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred must have matching output dimensions.")

    return {
        "r": _pairwise_corr(y_true, y_pred, copy=False),
        "r2": np.atleast_1d(np.asarray(r2_score(y_true, y_pred, multioutput="raw_values"), dtype=float)),
        "mae": np.atleast_1d(np.asarray(mean_absolute_error(y_true, y_pred, multioutput="raw_values"), dtype=float)),
        "rmse": np.atleast_1d(np.asarray(root_mean_squared_error(y_true, y_pred, multioutput="raw_values"), dtype=float)),
    }


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
    y_true = np.asarray(yte)
    y_pred = np.asarray(y_pred)

    out = {
        "y_true": y_true,
        "y_pred": y_pred,
        **prediction_metrics(y_true, y_pred)
    }

    for attr in dir(m):
        if not attr.endswith("_") or attr.startswith("__"):
            continue

        try: val = getattr(m, attr)
        except Exception: continue

        if isinstance(val, np.ndarray):
            out[attr] = val

        elif isinstance(val, (int, float, np.floating, np.integer)):
            out[attr] = val

    return out


def _set_optional_model_params(model, **params):
    available_params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    usable_params = {k: v for k, v in params.items() if k in available_params}

    if usable_params and hasattr(model, "set_params"):
        model.set_params(**usable_params)

    return model


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


def run_cv(X, y, model, split, C=None, residualize_X=False, param_grid=None, inner_split=None, grid_search_kw=None):
    """
    Cross-validated analysis with optional covariate residualization and scaling.

    Parameters

    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector.
    model : estimator
        sklearn-like estimator with fit and predict methods.
    split : CV splitter
        Any sklearn-like splitter implementing split(X).
    C : ndarray, optional
        Covariate matrix for residualization. If None, no residualization is performed.
    residualize_X : bool, default False
        Whether to residualize features in addition to the target. If True, X
        is residualized using the same covariates C and train/test split as y.

    Notes
    -----
    If the estimator exposes a ``fit_intercept`` parameter, it is
    explicitly set to ``True`` before fitting in each fold.
    """

    if grid_search_kw is None:
        grid_search_kw = {}

    res = []
    splits = list(split.split(X))
    for tr, te in splits:
        fold_res = {"fold": len(res) + 1}
        m = clone(model)
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        Ctr, Cte = (None, None) if C is None else (C[tr], C[te])

        if param_grid is not None:
            if inner_split is None:
                raise ValueError("param_grid provided without inner_split for hyperparameter tuning.")
            best_params, _, _ = grid_search(param_grid, Xtr, ytr, model, inner_split,
                                            C=Ctr, residualize_X=residualize_X, **grid_search_kw)
            m.set_params(**best_params)
            fold_res = fold_res | best_params

        if C is not None:
            Ctr, Cte = scale_splits((Ctr, Cte))[0]
            ytr, yte = residualize_splits(ytr, yte, Ctr, Cte, fit_intercept=True)
            if residualize_X:
                Xtr, Xte = residualize_splits(Xtr, Xte, Ctr, Cte, fit_intercept=True)
        Xtr, Xte = scale_splits((Xtr, Xte))[0]

        m = _set_optional_model_params(m, fit_intercept=True)
        m.fit(Xtr, ytr)
        res.append(fold_res | _eval_dict(m, Xte, yte))

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
            summary[f"mean_{k}"] = np.mean(arr, axis=0)
            summary[f"std_{k}"] = arr.std(axis=0)
        except Exception:
            continue

    return summary


def permutation_test(X, y, model, split, C=None, n_perm=1000, seed=0, metric="r", residualize_X=False,
                     agg_func=np.median, param_grid=None, inner_split=None, grid_search_kw=None):
    
    if grid_search_kw is None:
        grid_search_kw = {}

    rng = np.random.default_rng(seed)
    def score(y_): # will move outside later
        res = run_cv(X, y_, model, split, C=C, residualize_X=residualize_X,
                     param_grid=param_grid, inner_split=inner_split, grid_search_kw=grid_search_kw)
        fold_metric = np.asarray([d[metric] for d in res], dtype=float)
        return float(agg_func(fold_metric.reshape(-1)))

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


def grid_search(param_grid, X, y, model, split, C=None, metric="r", agg_func=np.mean, residualize_X=False):

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    results = []

    for combo in combos:
        params = dict(zip(keys, combo))

        m = clone(model).set_params(**params)

        res = run_cv(X, y, m, split, C=C, residualize_X=residualize_X)
        fold_metric = np.asarray([d[metric] for d in res], dtype=float)
        score = float(agg_func(fold_metric.reshape(-1)))

        results.append({
            "params": params,
            "score": score
        })

    best_func = np.argmax if metric in ["r", "r2"] else np.argmin
    best_idx = best_func([r["score"] for r in results])
    best_params = results[best_idx]["params"]
    best_score = results[best_idx]["score"]

    return best_params, best_score, results
