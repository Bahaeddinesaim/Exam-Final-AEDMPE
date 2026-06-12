"""Feature engineering for the tourism demand GOLD DATA."""

from __future__ import annotations

import numpy as np
import pandas as pd


FLIGHT_COST_MAP = {"low": 1.0, "medium": 2.0, "high": 3.0}
WEATHER_PENALTY_MAP = {"good": 0.0, "average": 0.25, "bad": 0.7}
SENTIMENT_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def minmax(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0-1 while handling constant vectors."""
    values = pd.to_numeric(series, errors="coerce")
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        return pd.Series(0.5, index=series.index)
    return (values - values.min()) / span


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aggregate noisy social reviews at country-destination level."""
    data = reviews.copy()
    data["sentiment_value"] = data["sentiment"].map(SENTIMENT_MAP)
    grouped = (
        data.groupby(["country", "destination"], dropna=False)
        .agg(
            review_count=("score", "size"),
            avg_review_score=("score", "mean"),
            sentiment_score=("sentiment_value", "mean"),
        )
        .reset_index()
    )
    return grouped


def aggregate_external(external: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external factors and encode weather/cost penalties."""
    data = external.copy()
    data["flight_cost_level"] = data["flight_price"].map(FLIGHT_COST_MAP)
    data["weather_penalty"] = data["weather"].map(WEATHER_PENALTY_MAP)
    return (
        data.groupby(["country", "destination"], dropna=False)
        .agg(
            flight_cost_level=("flight_cost_level", "mean"),
            weather_penalty=("weather_penalty", "mean"),
            bad_weather_share=("weather", lambda x: float((x == "bad").mean())),
        )
        .reset_index()
    )


def aggregate_campaigns(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate campaign records while exposing suspicious rows."""
    data = campaigns.copy()
    data["campaign_efficiency"] = np.where(
        data["campaign_budget"].gt(0),
        data["roi"] / data["campaign_budget"],
        np.nan,
    )
    return (
        data.groupby(["country", "destination"], dropna=False)
        .agg(
            historical_budget=("campaign_budget", "sum"),
            avg_conversion_rate=("conversion_rate", "mean"),
            avg_roi=("roi", "mean"),
            campaign_efficiency=("campaign_efficiency", "mean"),
            campaign_quality_alerts=("campaign_contradiction", "sum"),
        )
        .reset_index()
    )


def build_country_market_features(market: pd.DataFrame) -> pd.DataFrame:
    """Create country-level demand trend features from monthly signals."""
    ordered = market.sort_values(["country", "month"]).copy()
    ordered["demand_lag_1"] = ordered.groupby("country")["demand_index"].shift(1)
    ordered["demand_ma_3"] = (
        ordered.groupby("country")["demand_index"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    latest = ordered.groupby("country", as_index=False).tail(1).copy()
    latest["demand_growth"] = (latest["demand_index"] - latest["demand_lag_1"]) / latest["demand_lag_1"]
    return latest[["country", "demand_index", "demand_lag_1", "demand_ma_3", "demand_growth"]]


def build_gold_data(
    destinations: pd.DataFrame,
    market: pd.DataFrame,
    reviews: pd.DataFrame,
    external: pd.DataFrame,
    campaigns: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final country-destination GOLD DATA with documented scoring."""
    gold = destinations.copy()
    gold = gold.merge(aggregate_reviews(reviews), on=["country", "destination"], how="left")
    gold = gold.merge(aggregate_external(external), on=["country", "destination"], how="left")
    gold = gold.merge(aggregate_campaigns(campaigns), on=["country", "destination"], how="left")
    gold = gold.merge(build_country_market_features(market), on="country", how="left")

    defaults = {
        "review_count": 0,
        "avg_review_score": gold["rating"],
        "sentiment_score": 0,
        "flight_cost_level": 2,
        "weather_penalty": 0.25,
        "bad_weather_share": 0,
        "historical_budget": 0,
        "avg_conversion_rate": 0,
        "avg_roi": 0,
        "campaign_efficiency": 0,
        "campaign_quality_alerts": 0,
        "demand_growth": 0,
        "demand_index": gold["demand_index"].median(),
        "demand_lag_1": gold["demand_lag_1"].median(),
        "demand_ma_3": gold["demand_ma_3"].median(),
    }
    for column, default in defaults.items():
        gold[column] = gold[column].fillna(default)

    gold["quality_score"] = 0.55 * minmax(gold["rating"]) + 0.45 * minmax(gold["avg_review_score"])
    gold["tourism_score"] = (
        0.35 * minmax(gold["attractiveness"])
        + 0.25 * minmax(gold["visitors"])
        + 0.25 * minmax(gold["demand_index"])
        + 0.15 * minmax(gold["review_count"])
    )
    gold["attractiveness_cost_ratio"] = gold["attractiveness"] / gold["cost"].replace(0, np.nan)
    gold["forecasted_demand"] = (
        gold["visitors"]
        * (1 + gold["demand_growth"].clip(-0.25, 0.35))
        * (1 - 0.2 * gold["weather_penalty"])
    )
    gold["marketing_priority"] = (
        0.32 * minmax(gold["forecasted_demand"])
        + 0.22 * gold["quality_score"]
        + 0.18 * ((gold["sentiment_score"] + 1) / 2)
        + 0.16 * minmax(gold["attractiveness_cost_ratio"])
        + 0.12 * minmax(gold["campaign_efficiency"])
        - 0.18 * gold["weather_penalty"]
    ).clip(0, 1)
    gold["recommended_budget"] = (25000 + 75000 * gold["marketing_priority"]).round(-2)
    gold["business_rule_weather_ok"] = gold["weather_penalty"] < 0.65
    return gold.sort_values(["marketing_priority", "forecasted_demand"], ascending=False)
