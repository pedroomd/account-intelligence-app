import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from agents.brief_writer import MeetingBrief, BriefWriter
from utils.scoring import score_accounts

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "account_data.csv"

VIEWS = {
    "Top priority": {
        "sort_by": "priority_score",
        "ascending": False,
        "headline": "Top 10 accounts to work today",
        "score_col": "priority_score",
        "score_label": "Priority",
    },
    "Most likely to churn/reduce revenue": {
        "sort_by": "churn_risk_score",
        "ascending": False,
        "headline": "Top 10 churn-risk accounts",
        "score_col": "churn_risk_score",
        "score_label": "Churn Risk score",
    },
    "Most likely to expand": {
        "sort_by": "expansion_score",
        "ascending": False,
        "headline": "Top 10 expansion candidates",
        "score_col": "expansion_score",
        "score_label": "Expansion score",
    },
    "Renewals soon": {
        "sort_by": "days_to_next_renewal",
        "ascending": True,
        "headline": "Top 10 accounts with renewals approaching",
        "score_col": "days_to_next_renewal"
    },
}


@st.cache_data
def load_scored() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return score_accounts(df)


def render_table(df_top: pd.DataFrame, view: dict) -> None:
    display = pd.DataFrame({
        "Account": df_top["account_name"],
        "Segment": df_top["segment"],
        "Industry": df_top["industry"],
        "Revenue": df_top["current_revenue"].map(lambda v: f"${v:,.0f}"),
        "Renewal (days)": df_top["days_to_next_renewal"].astype(int),
        "Primary Signal": df_top["primary_signal"],
    })

    if "score_label" in view:
        display[view["score_label"]] = df_top[view["score_col"]].round(2)
    st.dataframe(display, hide_index=True, use_container_width=True)


def render_detail(row: pd.Series) -> None:
    st.subheader(row["account_name"])
    st.caption(
        f"{row['account_id']} · {row['segment']} · {row['industry']} · {row.get('region') or 'Region unknown'}"
    )
    if isinstance(row.get("account_description"), str):
        st.write(f":blue[{row['account_description']}]")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"${row['current_revenue']:,.0f}")
    util = row.get("seat_utilization")
    c2.metric(
        "Nr. Licensed Seats",
        f"{row['nr_licensed_seats']}" if pd.notna(row.get("nr_licensed_seats")) else "—"
    )
    c3.metric("Licensed Seats util.", f"{util:.0%}" if pd.notna(util) else "—")
    c4.metric("Open tickets", int(row["nr_support_tickets"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days to renewal", f"{int(row['days_to_next_renewal'])}d")
    c2.metric("Days since last touch",
              f"{int(row['days_since_last_sales_activity'])}d"
              if pd.notna(row.get("days_since_last_sales_activity"))
              else "Not available"
    )
    c3.metric(
        "AI adoption", f"{row['ai_usage']:.0%}" if pd.notna(row.get("ai_usage")) else "Not available"
    )
    action_score = row.get("action_score")
    primary_signal = row.get("primary_signal")
    c4.metric(f"{primary_signal} score", f"{action_score:.2f}" if pd.notna(action_score) else "—")

def build_brief_context(row: pd.Series) -> dict:
    transcript = row.get("call_transcript_summary", "")
    return {
        "account": {
            "name": row["account_name"],
            "segment": row["segment"],
            "industry": row["industry"],
            "description": row.get("account_description") or "",
        },
        "situation": {
            "status": row["status"],
            "churn_risk_score": round(float(row["churn_risk_score"]), 2),
            "expansion_score": round(float(row["expansion_score"]), 2),
            "renewal_in_days": int(row["days_to_next_renewal"]),
            "key_signals": list(row.get("reasons") or []),
        },
        "last_call": transcript if isinstance(transcript, str) and transcript.strip() else "No recent call recorded.",
    }


def render_brief(brief: MeetingBrief) -> None:
    st.markdown(f"**Situation.** {brief.situation}")

    st.markdown("**What changed**")
    for b in brief.what_changed:
        st.markdown(f"- {b}")

    st.markdown(f"**Recommended focus.** {brief.recommended_focus}")

    st.markdown("**Talking points**")
    for p in brief.talking_points:
        st.markdown(f"- {p}")

    st.markdown(f"**Careful about.** {brief.careful_about}")


def render_brief_section(row: pd.Series, brief_writer_agent: BriefWriter) -> None:
    st.markdown("#### Pre-meeting brief")

    account_id = row["account_id"]
    briefs: dict[str, MeetingBrief] = st.session_state.setdefault("briefs", {})

    col_btn, col_status = st.columns([1, 4])
    clicked = col_btn.button(
        "Regenerate brief" if account_id in briefs else "Generate brief",
        key=f"brief_{account_id}",
    )

    if clicked:
        context = build_brief_context(row)
        with st.spinner("Asking the model..."):
            try:
                briefs[account_id] = brief_writer_agent.generate_brief(context)
            except Exception as e:
                col_status.error(f"Couldn't generate brief: {e}")
                st.caption("Set GOOGLE_API_KEY in your environment and try again.")
                return

    if account_id in briefs:
        render_brief(briefs[account_id])
    else:
        st.caption("Click *Generate brief* to fuse the scores with the call transcript into a 30-second prep brief.")


def main() -> None:
    st.set_page_config(page_title="Account Intelligence", layout="wide")
    st.title("Account Intelligence")
    st.caption("Which accounts should I focus on today? What should I know before my next meeting?")

    scored = load_scored()

    with st.sidebar:
        st.header("View")
        view_name = st.radio("View", list(VIEWS.keys()), label_visibility="collapsed")
    view = VIEWS[view_name]

    top = scored.sort_values(view["sort_by"], ascending=view["ascending"]).head(10).reset_index(drop=True)

    st.subheader(view["headline"])
    render_table(top, view)

    st.divider()
    st.subheader("Account detail")
    options = top["account_name"].tolist()
    selected = st.selectbox("Pick an account from the list above", options, index=0)
    row = top[top["account_name"] == selected].iloc[0]
    render_detail(row)

    brief_writer_agent = BriefWriter()
    render_brief_section(row, brief_writer_agent)


if __name__ == "__main__":
    main()
