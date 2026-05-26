from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_DATA_PATH = Path("code/ml/ml_dataset.csv")


@dataclass(frozen=True)
class OLSResult:
    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    intercept_t: float
    slope_t: float
    intercept_pvalue: float
    slope_pvalue: float
    intercept_ci_low: float
    intercept_ci_high: float
    slope_ci_low: float
    slope_ci_high: float
    r2: float
    adj_r2: float
    mse: float
    rmse: float
    f_stat: float
    f_pvalue: float
    n_obs: int


@dataclass(frozen=True)
class MultiOLSResult:
    term_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]
    t_statistics: tuple[float, ...]
    p_values: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    r2: float
    adj_r2: float
    mse: float
    rmse: float
    f_stat: float
    f_pvalue: float
    n_obs: int
    df_model: int
    df_resid: int


def load_ml_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def prepare_regression_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"amas_score", "mseaq_anx", "math_hater_flg"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    frame = df.loc[:, ["amas_score", "mseaq_anx", "math_hater_flg"]].copy()
    frame["anxiety_score"] = frame["amas_score"].astype(float) + frame["mseaq_anx"].astype(float)
    frame["math_hater_flg"] = frame["math_hater_flg"].astype(int)
    return frame


def prepare_gender_regression_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"amas_score", "mseaq_anx", "math_hater_flg", "gender"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    frame = df.loc[:, ["amas_score", "mseaq_anx", "math_hater_flg", "gender"]].copy()
    frame["anxiety_score"] = frame["amas_score"].astype(float) + frame["mseaq_anx"].astype(float)
    frame["math_hater_flg"] = frame["math_hater_flg"].astype(int)

    gender = frame["gender"].astype(str).str.lower()
    frame["gender_group"] = np.where(
        gender.eq("man"),
        "man",
        np.where(gender.eq("woman"), "woman", "other"),
    )
    return frame


def fit_ols_from_matrix(X: np.ndarray, y: np.ndarray, term_names: tuple[str, ...]) -> MultiOLSResult:
    if X.ndim != 2:
        raise ValueError("X must be a 2D design matrix")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of rows")
    if X.shape[1] != len(term_names):
        raise ValueError("term_names must match the number of columns in X")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat

    n_obs, n_params = X.shape
    df_resid = n_obs - n_params
    if df_resid <= 0:
        raise ValueError("Need more observations than model parameters")

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_reg = ss_tot - ss_res
    mse = ss_res / df_resid
    rmse = float(np.sqrt(mse))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else np.nan)
    adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / df_resid if np.isfinite(r2) else np.nan

    xtx_inv = np.linalg.inv(X.T @ X)
    cov = mse * xtx_inv
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df_resid)
    crit = stats.t.ppf(0.975, df_resid)
    ci_low = beta - crit * se
    ci_high = beta + crit * se

    df_model = n_params - 1
    f_stat = float((ss_reg / df_model) / mse) if df_model > 0 else float("nan")
    f_pvalue = float(stats.f.sf(f_stat, df_model, df_resid)) if df_model > 0 else float("nan")

    return MultiOLSResult(
        term_names=term_names,
        coefficients=tuple(float(x) for x in beta),
        standard_errors=tuple(float(x) for x in se),
        t_statistics=tuple(float(x) for x in t_stats),
        p_values=tuple(float(x) for x in p_values),
        ci_low=tuple(float(x) for x in ci_low),
        ci_high=tuple(float(x) for x in ci_high),
        r2=float(r2),
        adj_r2=float(adj_r2),
        mse=float(mse),
        rmse=rmse,
        f_stat=f_stat,
        f_pvalue=f_pvalue,
        n_obs=n_obs,
        df_model=df_model,
        df_resid=df_resid,
    )


def fit_simple_ols(df: pd.DataFrame) -> OLSResult:
    frame = prepare_regression_frame(df)
    x = frame["math_hater_flg"].to_numpy(dtype=float)
    y = frame["anxiety_score"].to_numpy(dtype=float)

    fit = fit_ols_from_matrix(
        np.column_stack([np.ones_like(x), x]),
        y,
        ("Intercept", "math_hater_flg"),
    )

    return OLSResult(
        intercept=fit.coefficients[0],
        slope=fit.coefficients[1],
        intercept_se=fit.standard_errors[0],
        slope_se=fit.standard_errors[1],
        intercept_t=fit.t_statistics[0],
        slope_t=fit.t_statistics[1],
        intercept_pvalue=fit.p_values[0],
        slope_pvalue=fit.p_values[1],
        intercept_ci_low=fit.ci_low[0],
        intercept_ci_high=fit.ci_high[0],
        slope_ci_low=fit.ci_low[1],
        slope_ci_high=fit.ci_high[1],
        r2=fit.r2,
        adj_r2=fit.adj_r2,
        mse=fit.mse,
        rmse=fit.rmse,
        f_stat=fit.f_stat,
        f_pvalue=fit.f_pvalue,
        n_obs=fit.n_obs,
    )


def fit_math_hater_gender_ols(df: pd.DataFrame) -> MultiOLSResult:
    frame = prepare_gender_regression_frame(df)
    y = frame["anxiety_score"].to_numpy(dtype=float)
    predictors = pd.DataFrame(
        {
            "math_hater_flg": frame["math_hater_flg"].astype(float),
            "gender_man": (frame["gender_group"] == "man").astype(float),
            "gender_other": (frame["gender_group"] == "other").astype(float),
        }
    )
    X = np.column_stack([np.ones(len(predictors)), predictors.to_numpy(dtype=float)])
    return fit_ols_from_matrix(
        X,
        y,
        ("Intercept", "math_hater_flg", "gender_man", "gender_other"),
    )


def regression_summary_frame(result: OLSResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term": "Intercept",
                "estimate": result.intercept,
                "std_error": result.intercept_se,
                "t_stat": result.intercept_t,
                "p_value": result.intercept_pvalue,
                "ci_low": result.intercept_ci_low,
                "ci_high": result.intercept_ci_high,
            },
            {
                "term": "math_hater_flg",
                "estimate": result.slope,
                "std_error": result.slope_se,
                "t_stat": result.slope_t,
                "p_value": result.slope_pvalue,
                "ci_low": result.slope_ci_low,
                "ci_high": result.slope_ci_high,
            },
        ]
    )


def multi_regression_summary_frame(result: MultiOLSResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term": term,
                "estimate": est,
                "std_error": se,
                "t_stat": t,
                "p_value": p,
                "ci_low": low,
                "ci_high": high,
            }
            for term, est, se, t, p, low, high in zip(
                result.term_names,
                result.coefficients,
                result.standard_errors,
                result.t_statistics,
                result.p_values,
                result.ci_low,
                result.ci_high,
            )
        ]
    )


def regression_metrics_frame(result: OLSResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "n_obs", "value": result.n_obs},
            {"metric": "r2", "value": result.r2},
            {"metric": "adj_r2", "value": result.adj_r2},
            {"metric": "mse", "value": result.mse},
            {"metric": "rmse", "value": result.rmse},
            {"metric": "f_stat", "value": result.f_stat},
            {"metric": "f_pvalue", "value": result.f_pvalue},
        ]
    )


def multi_regression_metrics_frame(result: MultiOLSResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "n_obs", "value": result.n_obs},
            {"metric": "df_model", "value": result.df_model},
            {"metric": "df_resid", "value": result.df_resid},
            {"metric": "r2", "value": result.r2},
            {"metric": "adj_r2", "value": result.adj_r2},
            {"metric": "mse", "value": result.mse},
            {"metric": "rmse", "value": result.rmse},
            {"metric": "f_stat", "value": result.f_stat},
            {"metric": "f_pvalue", "value": result.f_pvalue},
        ]
    )


def gender_mean_sem_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = prepare_gender_regression_frame(df)
    summary = (
        frame.groupby(["gender_group", "math_hater_flg"], as_index=False)
        .agg(
            mean_anxiety=("anxiety_score", "mean"),
            sem_anxiety=("anxiety_score", "sem"),
            n=("anxiety_score", "size"),
        )
        .sort_values(["gender_group", "math_hater_flg"])
    )
    return summary


def add_gender_prediction_columns(frame: pd.DataFrame, result: MultiOLSResult) -> pd.DataFrame:
    out = prepare_gender_regression_frame(frame)
    out["predicted_anxiety"] = (
        result.coefficients[0]
        + result.coefficients[1] * out["math_hater_flg"].astype(float)
        + result.coefficients[2] * (out["gender_group"] == "man").astype(float)
        + result.coefficients[3] * (out["gender_group"] == "other").astype(float)
    )
    out["residual"] = out["anxiety_score"] - out["predicted_anxiety"]
    return out


def add_prediction_columns(frame: pd.DataFrame, result: OLSResult) -> pd.DataFrame:
    out = frame.copy()
    out["predicted_anxiety"] = result.intercept + result.slope * out["math_hater_flg"]
    out["residual"] = out["anxiety_score"] - out["predicted_anxiety"]
    return out
