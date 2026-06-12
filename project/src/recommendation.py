"""Business recommendation engine for destination promotion."""

from __future__ import annotations

import pandas as pd


def rank_destinations(gold: pd.DataFrame, max_per_country: int = 5, total_budget: float = 350_000) -> pd.DataFrame:
    """Apply business constraints and return the final promotion ranking."""
    eligible = gold[gold["business_rule_weather_ok"]].copy()
    eligible = eligible.sort_values(["country", "marketing_priority"], ascending=[True, False])
    eligible["rank_country"] = eligible.groupby("country").cumcount() + 1
    eligible = eligible[eligible["rank_country"] <= max_per_country].copy()
    eligible = eligible.sort_values("marketing_priority", ascending=False)
    eligible["budget_weight"] = eligible["marketing_priority"] / eligible["marketing_priority"].sum()
    eligible["allocated_budget"] = (eligible["budget_weight"] * total_budget).round(-2)
    eligible["recommendation_reason"] = eligible.apply(
        lambda row: (
            f"Demand forecast={row['forecasted_demand']:.0f}, quality={row['quality_score']:.2f}, "
            f"sentiment={row['sentiment_score']:.2f}, weather penalty={row['weather_penalty']:.2f}"
        ),
        axis=1,
    )
    return eligible[
        [
            "country",
            "destination",
            "destination_original",
            "rank_country",
            "marketing_priority",
            "forecasted_demand",
            "quality_score",
            "sentiment_score",
            "weather_penalty",
            "campaign_efficiency",
            "allocated_budget",
            "recommendation_reason",
        ]
    ]
