import pandas as pd

CHURN_WEIGHTS = {"util": 0.35, "activity": 0.30, "tickets": 0.20, "ai": 0.15}
EXPANSION_WEIGHTS = {"util": 0.30, "activity": 0.30, "tickets": 0.25, "ai": 0.15}

COMPETITOR_PATTERN = r"competitor|replace|evaluating|switch"
FRUSTRATION_PATTERN = r"frustrat|delay|complain|escalat|unhappy|concern|disappoint"
EXPANSION_PATTERN = r"seat|expand|upsell|additional|roll out|rollout|fast-track"


def score_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk and expansion scores, as well as priority scores and labels.
    """
    df = df.copy()

    df["seat_utilization"] = df["nr_active_users"] / df["nr_licensed_seats"]
    df["no_recent_call"] = df["call_transcript_summary"].isna().astype(int)

    txt = df["call_transcript_summary"].fillna("").str.lower()
    df["t_competitor"] = txt.str.contains(COMPETITOR_PATTERN, regex=True).astype(int)
    df["t_frustration"] = txt.str.contains(FRUSTRATION_PATTERN, regex=True).astype(int)
    df["t_expansion"] = txt.str.contains(EXPANSION_PATTERN, regex=True).astype(int)

    days_last_activity_norm = df["days_since_last_sales_activity"].fillna(df["days_since_last_sales_activity"].median())
    tickets_norm = df["nr_support_tickets"].fillna(df["nr_support_tickets"].median())

    df["activity_norm"] = days_last_activity_norm.rank(pct=True)
    df["tickets_norm"] = tickets_norm.rank(pct=True)

    ai = df["ai_usage"].fillna(df["ai_usage"].median())
    util = df["seat_utilization"].fillna(df["seat_utilization"].median())
    activity = df["activity_norm"].fillna(df["activity_norm"].median())
    tickets = df["tickets_norm"].fillna(df["tickets_norm"].median())

    df["churn_risk_score"] = (
        CHURN_WEIGHTS["util"] * (1 - util)
        + CHURN_WEIGHTS["activity"] * activity
        + CHURN_WEIGHTS["tickets"] * tickets
        + CHURN_WEIGHTS["ai"] * (1 - ai)
        + 0.15 * df["t_competitor"]
        + 0.10 * df["t_frustration"]
    ).clip(0, 1)

    df["expansion_score"] = (
        EXPANSION_WEIGHTS["util"] * util
        + EXPANSION_WEIGHTS["activity"] * (1 - activity)
        + EXPANSION_WEIGHTS["tickets"] * (1 - tickets)
        + EXPANSION_WEIGHTS["ai"] * ai
        + 0.10 * df["t_expansion"]
    ).clip(0, 1)

    df["action_score"] = df[["churn_risk_score", "expansion_score"]].max(axis=1)
    df["revenue_norm"] = df["current_revenue"].rank(pct=True)
    df["urgency"] = 1 - df["days_to_next_renewal"].rank(pct=True)

    df["priority_score"] = df["revenue_norm"] * df["urgency"] * df["action_score"]
    assert df["priority_score"].notna().all()

    df["primary_signal"] = df.apply(
        lambda row: "Churn risk" if row["churn_risk_score"] >= row["expansion_score"] else "Expansion",
        axis=1
    )
    df["status"] = df.apply(_llm_status, axis=1)

    key_reasons_thresholds = {
        "tickets_high": df["nr_support_tickets"].quantile(0.75).round(1),
        "last_sales_activity_high": 90,
        "seat_utilization_low": df["seat_utilization"].quantile(0.25).round(1),
        "seat_utilization_high": df["seat_utilization"].quantile(0.75).round(1),
        "ai_usage_low": df["ai_usage"].quantile(0.25).round(1),
        "ai_usage_high": df["ai_usage"].quantile(0.75).round(1),
        "days_to_next_renewal_low": 60

    }
    df["reasons"] = df.apply(_key_reasons, axis=1, thresholds=key_reasons_thresholds)

    return df

def _llm_status(row: pd.Series) -> str:
    """
    Translate raw scores into human-friendly labels for the UI and for the LLM context.
    """
    churn_elevated = row["churn_risk_score"] >= 0.5
    expand_elevated = row["expansion_score"] >= 0.5
    if churn_elevated and expand_elevated:
        return "risk of churning and expanding at the same time"
    if churn_elevated:
        return "risk of churning"
    if expand_elevated:
        return "expanding"
    return "steady, no strong signals of churn or expansion"


def _key_reasons(row: pd.Series, thresholds: dict) -> list[str]:
    """Plain-English signals for the UI and for the LLM context.

    Each entry includes enough raw numbers that the brief writer can reference
    them verbatim instead of paraphrasing percentiles.
    """
    reasons: list[str] = []

    seat_util = row.get("seat_utilization")
    active = row.get("nr_active_users")
    licensed = row.get("nr_licensed_seats")

    if pd.notna(seat_util) and pd.notna(active) and pd.notna(licensed):
        seat_util_usage = None
        if seat_util <= thresholds["seat_utilization_low"]:
            seat_util_usage = "low"
        elif seat_util >= thresholds["seat_utilization_high"]:
            seat_util_usage = "high"

        if seat_util_usage:
            reasons.append(
                f"Seat utilization {seat_util_usage}: {int(active)} of {int(licensed)} licensed seats active"
            )

    ai_usage = row.get("ai_usage")
    if pd.notna(ai_usage):
        if ai_usage <= thresholds["ai_usage_low"]:
            reasons.append(f"AI features barely adopted ({ai_usage:.0%})")
        elif ai_usage >= thresholds["ai_usage_high"]:
            reasons.append(f"Strong AI adoption ({ai_usage:.0%})")

    tickets = row.get("nr_support_tickets")
    if pd.notna(tickets) and tickets > thresholds["tickets_high"]:
        reasons.append(f"{int(tickets)} open support tickets")

    days_activity = row.get("days_since_last_sales_activity")
    if pd.notna(days_activity) and days_activity >= thresholds["last_sales_activity_high"]:
        reasons.append(f"No sales contact in {int(days_activity)} days")

    days_renewal = row.get("days_to_next_renewal")
    if pd.notna(days_renewal) and days_renewal <= thresholds["days_to_next_renewal_low"]:
        reasons.append(f"Renewal in {int(days_renewal)} days")

    if row.get("t_competitor"):
        reasons.append("Competitor mentioned on last call")
    if row.get("t_frustration"):
        reasons.append("Frustration signals on last call")
    if row.get("t_expansion"):
        reasons.append("Expansion intent on last call")

    return reasons[:6]
