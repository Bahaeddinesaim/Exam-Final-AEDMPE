"""Forecasting baselines and models for monthly tourism demand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class ModelResult:
    model: str
    mae: float
    rmse: float
    mape: float
    r2: float


def add_time_features(market: pd.DataFrame) -> pd.DataFrame:
    """Create lagged features with no future leakage."""
    df = market.sort_values(["country", "month"]).copy()
    df["month_num"] = df["month"].dt.month
    df["year"] = df["month"].dt.year
    df["time_index"] = df.groupby("country").cumcount()
    df["lag_1"] = df.groupby("country")["demand_index"].shift(1)
    df["lag_3"] = df.groupby("country")["demand_index"].shift(3)
    df["ma_3"] = df.groupby("country")["demand_index"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["country_code"] = df["country"].astype("category").cat.codes
    return df.dropna(subset=["lag_1"])


def temporal_split(df: pd.DataFrame, test_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so the model is evaluated on future months only."""
    cutoff = df["month"].max() - pd.DateOffset(months=test_months - 1)
    train = df[df["month"] < cutoff].copy()
    test = df[df["month"] >= cutoff].copy()
    return train, test


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> ModelResult:
    """Compute forecasting metrics with MAPE guarded against zero targets."""
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    return ModelResult(
        model=model_name,
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        mape=float(np.nan_to_num(mape)),
        r2=float(r2_score(y_true, y_pred)),
    )


def train_and_evaluate(market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train baselines and two ML models using a temporal holdout."""
    df = add_time_features(market)
    train, test = temporal_split(df)
    features = ["country_code", "month_num", "year", "time_index", "lag_1", "lag_3", "ma_3"]
    y_train = train["demand_index"]
    y_test = test["demand_index"]
    train[features] = train[features].fillna({"lag_3": train["lag_1"], "ma_3": train["lag_1"]})
    test[features] = test[features].fillna({"lag_3": test["lag_1"], "ma_3": test["lag_1"]})
    train[features] = train[features].fillna(train[features].median(numeric_only=True))
    test[features] = test[features].fillna(train[features].median(numeric_only=True))

    predictions = pd.DataFrame({"country": test["country"], "month": test["month"], "actual": y_test})
    predictions["baseline_persistence"] = test["lag_1"]
    predictions["baseline_moving_average"] = test["ma_3"]

    models = {
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    for name, model in models.items():
        model.fit(train[features], y_train)
        predictions[name] = model.predict(test[features])

    results = [
        evaluate_predictions(y_test, predictions["baseline_persistence"].to_numpy(), "Baseline A - naive persistence"),
        evaluate_predictions(y_test, predictions["baseline_moving_average"].to_numpy(), "Baseline B - moving average"),
        evaluate_predictions(y_test, predictions["random_forest"].to_numpy(), "Random Forest"),
        evaluate_predictions(y_test, predictions["gradient_boosting"].to_numpy(), "Gradient Boosting"),
    ]
    return pd.DataFrame([r.__dict__ for r in results]).sort_values("rmse"), predictions


def evaluate_model(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    approach_type: str = "Forecasting model",
) -> dict[str, Any]:
    """Evaluate one model with metrics used consistently across baselines and algorithms."""
    truth = pd.Series(y_true).astype(float)
    pred = pd.Series(y_pred, index=truth.index).astype(float)
    valid = truth.notna() & pred.notna()
    if valid.sum() == 0:
        return {
            "Model": model_name,
            "Approach type": approach_type,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            "Business interpretation": "Model unavailable or no valid predictions.",
            "Status": "Unavailable",
        }
    truth = truth[valid]
    pred = pred[valid]
    denominator = truth.replace(0, np.nan)
    mape = (np.abs((truth - pred) / denominator).replace([np.inf, -np.inf], np.nan).mean()) * 100
    rmse = mean_squared_error(truth, pred) ** 0.5
    mae = mean_absolute_error(truth, pred)
    r2 = r2_score(truth, pred) if len(truth) > 1 else np.nan
    return {
        "Model": model_name,
        "Approach type": approach_type,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(np.nan_to_num(mape)),
        "R2": float(np.nan_to_num(r2, nan=np.nan)),
        "Business interpretation": _business_interpretation(model_name, rmse),
        "Status": "OK",
    }


def _business_interpretation(model_name: str, rmse: float) -> str:
    """Translate statistical performance into a business-facing interpretation."""
    if np.isnan(rmse):
        return "Unavailable due to dependency or data limitation."
    if "Baseline" in model_name:
        return "Reference point used to prove whether advanced AI adds value."
    return "Candidate model for country demand forecasting and marketing prioritization."


def prepare_modeling_dataset(gold: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Build a clean country-period table for supervised temporal forecasting.

    The available real temporal signal is `demand_index` by country and month.
    Destination-level visitors are static, so they are aggregated by country and used
    as contextual explanatory variables, not as a time-varying target.
    """
    temporal = market.rename(columns={"month": "period"}).copy()
    temporal["period"] = pd.to_datetime(temporal["period"], errors="coerce")
    temporal = temporal.dropna(subset=["country", "period", "demand_index"])
    temporal = temporal.sort_values(["country", "period"])

    country_features = (
        gold.groupby("country", as_index=False)
        .agg(
            visitors=("visitors", "sum"),
            attractiveness=("attractiveness", "mean"),
            cost=("cost", "mean"),
            rating=("rating", "mean"),
            sentiment_score=("sentiment_score", "mean"),
            weather_score=("weather_penalty", "mean"),
            campaign_budget=("historical_budget", "sum"),
            marketing_priority=("marketing_priority", "mean"),
        )
    )
    modeling_df = temporal.merge(country_features, on="country", how="left")
    modeling_df["market_signal"] = modeling_df["demand_index"]
    modeling_df["target_next_period"] = modeling_df.groupby("country")["demand_index"].shift(-1)
    for lag in [1, 2, 3, 7, 14, 30]:
        modeling_df[f"lag_{lag}"] = modeling_df.groupby("country")["demand_index"].shift(lag)

    shifted = modeling_df.groupby("country")["demand_index"].shift(1)
    for window in [3, 7, 14, 30]:
        modeling_df[f"rolling_mean_{window}"] = (
            shifted.groupby(modeling_df["country"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
    for window in [7, 30]:
        modeling_df[f"rolling_std_{window}"] = (
            shifted.groupby(modeling_df["country"]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        )
    modeling_df["rolling_min_30"] = (
        shifted.groupby(modeling_df["country"]).rolling(30, min_periods=1).min().reset_index(level=0, drop=True)
    )
    modeling_df["rolling_max_30"] = (
        shifted.groupby(modeling_df["country"]).rolling(30, min_periods=1).max().reset_index(level=0, drop=True)
    )
    modeling_df["month_num"] = modeling_df["period"].dt.month
    modeling_df["month"] = modeling_df["period"].dt.month
    modeling_df["quarter"] = modeling_df["period"].dt.quarter
    modeling_df["year"] = modeling_df["period"].dt.year
    modeling_df["time_index"] = modeling_df.groupby("country").cumcount()

    numeric_columns = [
        "visitors",
        "attractiveness",
        "cost",
        "rating",
        "sentiment_score",
        "weather_score",
        "campaign_budget",
        "marketing_priority",
        "market_signal",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_std_7",
        "rolling_std_30",
        "rolling_min_30",
        "rolling_max_30",
    ]
    for column in numeric_columns:
        modeling_df[column] = pd.to_numeric(modeling_df[column], errors="coerce")
        modeling_df[column] = modeling_df[column].fillna(modeling_df[column].median())
    return modeling_df.dropna(subset=["target_next_period"]).reset_index(drop=True)


def temporal_train_test_split(modeling_df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by chronological periods; no random split is allowed for forecasting."""
    periods = sorted(modeling_df["period"].dropna().unique())
    split_idx = max(1, int(len(periods) * (1 - test_ratio)))
    cutoff = periods[split_idx]
    train = modeling_df[modeling_df["period"] < cutoff].copy()
    test = modeling_df[modeling_df["period"] >= cutoff].copy()
    return train, test


def build_sklearn_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create a reusable preprocessing layer for structured forecasting models."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _train_boosting_model(preprocessor: ColumnTransformer) -> tuple[str, Pipeline]:
    """Use XGBoost when installed; otherwise fall back to sklearn Gradient Boosting."""
    try:
        from xgboost import XGBRegressor

        return (
            "XGBoost",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=250,
                            learning_rate=0.05,
                            max_depth=3,
                            objective="reg:squarederror",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        )
    except ImportError:
        return (
            "Gradient Boosting",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", GradientBoostingRegressor(random_state=42)),
                ]
            ),
        )


def sarimax_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Forecast per country with SARIMAX while never breaking the pipeline."""
    predictions = pd.Series(np.nan, index=test.index, dtype=float)
    limitations = []
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        limitations.append({"country": "ALL", "issue": "statsmodels is not installed; SARIMAX skipped."})
        return predictions, pd.DataFrame(limitations)

    for country, country_test in test.groupby("country"):
        country_train = train[train["country"] == country].sort_values("period")
        if len(country_train) < 8 or len(country_test) == 0:
            limitations.append({"country": country, "issue": "Insufficient history for SARIMAX."})
            continue
        seasonal_order = (1, 1, 1, 12) if len(country_train) >= 30 else (0, 0, 0, 0)
        try:
            series = country_train.set_index("period")["target_next_period"].astype(float)
            series = series.asfreq("MS")
            exog_cols = [
                "attractiveness",
                "cost",
                "rating",
                "sentiment_score",
                "weather_score",
                "campaign_budget",
                "market_signal",
            ]
            train_exog = country_train.set_index("period")[exog_cols].astype(float).asfreq("MS").ffill().bfill()
            test_exog = country_test.sort_values("period").set_index("period")[exog_cols].astype(float).asfreq("MS").ffill().bfill()
            model = SARIMAX(
                series,
                exog=train_exog,
                order=(1, 1, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=len(country_test), exog=test_exog)
            predictions.loc[country_test.index] = forecast.to_numpy()
        except Exception as exc:  # noqa: BLE001 - explicit notebook limitation table
            limitations.append({"country": country, "issue": f"SARIMAX failed: {exc}"})
    return predictions, pd.DataFrame(limitations)


def prophet_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Forecast per country with Prophet when available; otherwise return a documented limitation."""
    predictions = pd.Series(np.nan, index=test.index, dtype=float)
    limitations = []
    try:
        from prophet import Prophet
    except ImportError:
        limitations.append({"country": "ALL", "issue": "prophet is not installed; Prophet skipped gracefully."})
        return predictions, pd.DataFrame(limitations)

    for country, country_test in test.groupby("country"):
        country_train = train[train["country"] == country].sort_values("period")
        if len(country_train) < 8:
            limitations.append({"country": country, "issue": "Insufficient history for Prophet."})
            continue
        yearly = len(country_train) >= 24
        regressor_cols = ["attractiveness", "cost", "rating", "sentiment_score", "weather_score", "campaign_budget"]
        prophet_df = country_train[["period", "target_next_period"] + regressor_cols].rename(
            columns={"period": "ds", "target_next_period": "y"}
        )
        future = country_test[["period"] + regressor_cols].rename(columns={"period": "ds"})
        try:
            model = Prophet(
                yearly_seasonality=yearly,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="additive",
            )
            for regressor in regressor_cols:
                model.add_regressor(regressor)
            model.fit(prophet_df)
            forecast = model.predict(future)
            predictions.loc[country_test.index] = forecast["yhat"].to_numpy()
        except Exception as exc:  # noqa: BLE001 - explicit notebook limitation table
            limitations.append({"country": country, "issue": f"Prophet failed: {exc}"})
    return predictions, pd.DataFrame(limitations)


def extract_feature_importance(model: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    """Extract feature importance from a fitted tree-based sklearn pipeline."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    preprocessor = model.named_steps["preprocessor"]
    feature_names = list(numeric_features)
    try:
        cat_names = preprocessor.named_transformers_["cat"].named_steps["one_hot"].get_feature_names_out(categorical_features)
        feature_names.extend(cat_names)
    except Exception:
        feature_names.extend(categorical_features)
    return (
        pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def time_series_cv_scores(
    modeling_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate ML forecasting models with chronological TimeSeriesSplit."""
    ordered = modeling_df.sort_values(["period", "country"]).reset_index(drop=True)
    features = numeric_features + categorical_features
    target = ordered["target_next_period"]
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    model_factories = {
        "Random Forest + lag features": lambda: Pipeline(
            steps=[
                ("preprocessor", build_sklearn_preprocessor(numeric_features, categorical_features)),
                ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42)),
            ]
        ),
    }
    boosting_name, _ = _train_boosting_model(build_sklearn_preprocessor(numeric_features, categorical_features))
    model_factories[f"{boosting_name} + lag features"] = lambda: _train_boosting_model(
        build_sklearn_preprocessor(numeric_features, categorical_features)
    )[1]

    for fold, (train_idx, test_idx) in enumerate(tscv.split(ordered), start=1):
        train_fold = ordered.iloc[train_idx]
        test_fold = ordered.iloc[test_idx]
        for model_name, factory in model_factories.items():
            model = factory()
            model.fit(train_fold[features], target.iloc[train_idx])
            y_pred = model.predict(test_fold[features])
            metrics = evaluate_model(
                target.iloc[test_idx],
                y_pred,
                model_name,
                "TimeSeriesSplit supervised forecasting",
            )
            rows.append(
                {
                    "Model": model_name,
                    "Fold": fold,
                    "Train start": train_fold["period"].min(),
                    "Train end": train_fold["period"].max(),
                    "Test start": test_fold["period"].min(),
                    "Test end": test_fold["period"].max(),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE": metrics["MAPE"],
                }
            )

    fold_results = pd.DataFrame(rows)
    summary = (
        fold_results.groupby("Model", as_index=False)
        .agg(
            Mean_MAE=("MAE", "mean"),
            Std_MAE=("MAE", "std"),
            Mean_RMSE=("RMSE", "mean"),
            Std_RMSE=("RMSE", "std"),
            Mean_MAPE=("MAPE", "mean"),
            Std_MAPE=("MAPE", "std"),
        )
        .sort_values("Mean_RMSE")
    )
    return fold_results, summary


def train_full_model_suite(
    gold: pd.DataFrame,
    market: pd.DataFrame,
    reports_dir: Path,
    gold_dir: Path,
    models_dir: Path,
) -> dict[str, pd.DataFrame | str]:
    """Train the complete Part F model suite and persist professional deliverables."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    modeling_df = prepare_modeling_dataset(gold, market)
    train, test = temporal_train_test_split(modeling_df)
    numeric_features = [
        "visitors",
        "attractiveness",
        "cost",
        "rating",
        "sentiment_score",
        "weather_score",
        "campaign_budget",
        "marketing_priority",
        "market_signal",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_std_7",
        "rolling_std_30",
        "rolling_min_30",
        "rolling_max_30",
        "month",
        "month_num",
        "quarter",
        "year",
        "time_index",
    ]
    categorical_features = ["country"]
    features = numeric_features + categorical_features
    y_train = train["target_next_period"]
    y_test = test["target_next_period"]

    predictions = test[["country", "period", "demand_index", "target_next_period"]].copy()
    predictions = predictions.rename(columns={"target_next_period": "actual"})
    predictions["month"] = predictions["period"]
    predictions["baseline_persistence"] = test["market_signal"]
    predictions["baseline_moving_average"] = test["rolling_mean_3"]

    preprocessor = build_sklearn_preprocessor(numeric_features, categorical_features)
    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42)),
        ]
    )
    random_forest.fit(train[features], y_train)
    predictions["random_forest"] = random_forest.predict(test[features])

    boosting_name, boosting_model = _train_boosting_model(build_sklearn_preprocessor(numeric_features, categorical_features))
    boosting_model.fit(train[features], y_train)
    predictions["gradient_boosting"] = boosting_model.predict(test[features])
    fitted_sklearn_models = {
        "Random Forest": random_forest,
        boosting_name: boosting_model,
    }

    sarimax_pred, sarimax_limits = sarimax_predictions(train, test)
    prophet_pred, prophet_limits = prophet_predictions(train, test)
    predictions["sarimax"] = sarimax_pred
    predictions["prophet"] = prophet_pred

    performance = pd.DataFrame(
        [
            evaluate_model(y_test, predictions["baseline_persistence"], "Baseline Persistence", "Temporal baseline"),
            evaluate_model(y_test, predictions["baseline_moving_average"], "Baseline Moving Average", "Temporal baseline"),
            evaluate_model(y_test, predictions["sarimax"], "SARIMAX", "Statistical time-series model"),
            evaluate_model(y_test, predictions["prophet"], "Prophet", "Additive time-series model"),
            evaluate_model(y_test, predictions["random_forest"], "Random Forest + lag features", "Supervised forecasting with lag features"),
            evaluate_model(y_test, predictions["gradient_boosting"], f"{boosting_name} + lag features", "Supervised forecasting with lag features"),
        ]
    ).sort_values("RMSE", na_position="last")

    best_model = performance[performance["Status"] == "OK"].iloc[0]["Model"]
    best_row = performance[performance["Status"] == "OK"].iloc[0]
    model_column_map = {
        "Baseline Persistence": "baseline_persistence",
        "Baseline Moving Average": "baseline_moving_average",
        "SARIMAX": "sarimax",
        "Prophet": "prophet",
        "Random Forest + lag features": "random_forest",
        f"{boosting_name} + lag features": "gradient_boosting",
    }
    best_prediction_column = model_column_map[str(best_model)]
    best_summary = (
        f"# Best Model Summary\n\n"
        f"Le modele retenu est **{best_model}** car il obtient la meilleure performance RMSE "
        f"sur l'ensemble de test temporel: MAE={best_row['MAE']:.2f}, "
        f"RMSE={best_row['RMSE']:.2f}, MAPE={best_row['MAPE']:.2f}% et "
        f"R2={best_row['R2']:.2f}. Il doit etre lu avec prudence: la performance "
        "statistique ne remplace pas l'analyse de robustesse, l'interpretabilite, la qualite "
        "des donnees et la validation metier marketing.\n\n"
        "Le split temporel respecte l'ordre chronologique des observations et evite la fuite "
        "de donnees. Les modeles Random Forest et XGBoost/Gradient Boosting ont ete adaptes "
        "au forecasting via la creation de variables de retard et de moyennes glissantes. "
        "Ils ne sont donc pas utilises comme simples modeles tabulaires, mais comme approches "
        "de prevision supervisee temporelle.\n\n"
        "Une validation croisee temporelle TimeSeriesSplit a egalement ete appliquee aux "
        "modeles ML afin de mesurer leur robustesse sur plusieurs fenetres chronologiques "
        "successives sans fuite d'information.\n"
    )

    feature_importance = extract_feature_importance(random_forest, numeric_features, categorical_features)
    cv_results, cv_summary = time_series_cv_scores(modeling_df, numeric_features, categorical_features)
    limitations = pd.concat(
        [
            sarimax_limits.assign(model="SARIMAX"),
            prophet_limits.assign(model="Prophet"),
        ],
        ignore_index=True,
    )

    modeling_df.to_csv(reports_dir / "modeling_dataset.csv", index=False)
    performance.to_csv(reports_dir / "model_performance.csv", index=False)
    predictions.to_csv(gold_dir / "forecast_predictions.csv", index=False)
    predictions.to_csv(reports_dir / "forecast_predictions.csv", index=False)
    feature_importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    cv_results.to_csv(reports_dir / "time_series_cv_results.csv", index=False)
    cv_summary.to_csv(reports_dir / "time_series_cv_summary.csv", index=False)
    limitations.to_csv(reports_dir / "time_series_model_limits.csv", index=False)
    (reports_dir / "best_model_summary.md").write_text(best_summary, encoding="utf-8")
    country_forecast = (
        predictions.sort_values("period")
        .groupby("country", as_index=False)
        .tail(1)[["country", best_prediction_column]]
        .rename(columns={best_prediction_column: "forecast_next_period"})
    )
    gold_with_forecast = gold.merge(country_forecast, on="country", how="left")
    gold_with_forecast.to_csv(gold_dir / "gold_tourism_data_with_forecast.csv", index=False)

    persisted_model_key = str(best_model).replace(" + lag features", "")
    model_to_persist = fitted_sklearn_models.get(persisted_model_key, random_forest)
    joblib.dump(model_to_persist, models_dir / "best_model.pkl")

    return {
        "modeling_df": modeling_df,
        "performance": performance,
        "predictions": predictions,
        "feature_importance": feature_importance,
        "cv_results": cv_results,
        "cv_summary": cv_summary,
        "limitations": limitations,
        "best_model": str(best_model),
    }
