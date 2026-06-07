# Account Intelligence App

A small app that helps an Account Executive answer two questions: which of my accounts should I focus on, and what do I need to know before my next meeting?

## What it does
The app has four views, each surfacing a different slice of the account book:

- Priority (default) - the top 10 highest-priority accounts, ranked by a combined priority score (explained below).
- Churn risk - the top 10 accounts most likely to churn or reduce their spend.
- Expansion - the top 10 accounts most likely to expand
- Renewals soon - the top 10 accounts with the nearest contract renewals.

Within any view, you can select an account from the table to see its full detail and generate an AI-powered pre-meeting brief - a 30-second read for the AE, produced by an LLM from the account's signals and most recent call summary.

The four views exist because there isn't one "correct" way to prioritise — it depends on the domain knowledge and what the AE is trying to do that day (save an account, find upsell, prepare for a renewal).

## Setup and running

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run streamlit run src/app.py
```

The pre-meeting brief uses an LLM, so you'll need to set an API key:

```bash
export GOOGLE_API_KEY=your_key_here
```

## App architecture
The UI app is built with Streamlit, and the data is in the /data directory.

Inside src/ directory, you'll find:
- `app.py`: the main Streamlit app, which defines the UI and interactions.
- `eda.ipynb`: a Jupyter notebook where I explored the data, engineered features and developed the scoring logic for the priority view.
- `agents/`: a directory containing the code for the AI agents. Despite two agents being defined, only the pre-meeting brief agent is currently used in the app. The other agent is just there for demonstration purposes, to show how we could leverage an LLM to enrich the data with more features/signals based on the call_transcript_summary column.
- `utils/`: a directory for utility functions, where `scoring.py` lives, which contains the logic for calculating the priority score and churn/expansion risk scores.

## Key decisions
I started by exploring the dataset to figure out which features/signals had the most impact on revenue decreasing (churn risk) and revenue increasing (expansion). Then I built a `churn risk score` and a `expansion score` based on those features, with more or less arbitrary weights roughly taken from the correlations. A better approach here would clearly have been to distribute the importance of each feature through a predictive model, but this works for demonstration purposes.

I also noticed there were a lot of generated words in the call_transcript_summary column that clearly had an influence on whether an account might reduce or increase its revenue. I know this is just a synthetic dataset and the transcripts were probably generated from the outcome but still, it felt like a missed opportunity not to leverage that information in the scores. So I built a simple keyword search to identify whether the summary contained positive or negative signals, and used them as a boost for the final scores. Ideally this is another area where an LLM could be leveraged to extract more meaningful insights from the call summaries, rather than just a binary positive/negative keyword flag.

In the end, the final churn risk is defined like this:

```
churn_risk_score = w1·(1 − seat_utilization)
                 + w2·rank(days_since_last_sales_activity)
                 + w3·rank(nr_support_tickets)
                 + w4·(1 − ai_usage)
                 + capped transcript boost
```

And the expansion score is defined like this:

```
expansion_score = w1·seat_utilization
                + w2·rank(days_since_last_sales_activity)
                + w3·rank(nr_support_tickets)
                + w4·ai_usage
                + capped transcript boost
```

For the priority score, the idea is different — the way I see it there's no right/wrong answer to validate against, since it's more of a business judgment than something that actually happened in the data. I framed it as revenue_at_stake × urgency × max(churn_score, expansion_score): how much money is on the line, how soon the renewal is, and the account's strongest reason to act. Revenue and renewal timing come in as multipliers rather than as risk signals, since they don't really predict whether an account goes up or down — only how much is at stake and how urgent it is.
