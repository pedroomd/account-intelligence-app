import asyncio
from typing import Literal
from pydantic_ai import Agent
from pydantic import BaseModel, Field

# Note: this agent is just for demonstration purposes and is not actually used in the app.
# The signals it extracts are not included in the scoring function or the brief context.
# But it serves as an example of how an llm could be used in to extract structured signals
# from a text column data (in this case, call transcript summaries).

class TranscriptParserOutput(BaseModel):

    expansion_intention: bool = Field(
        default=False,
        description="Whether the customer expressed intention to expand their usage of the product."
    )

    churn_risk_signal: Literal[0, 1, 2] = Field(
        default=0,
        ge=0, le=2,
        description=(
            "Level of churn risk expressed in the call summary. "
            "2: the customer clearly expresses dissatisfaction with the product, states "
            "an intention to stop or significantly reduce usage, says they are evaluating or "
            "switching to a competitor/alternative approaches, or raises a serious blocker to renewal "
            "(e.g. budget cuts, restructuring, loss of executive sponsor, etc).\n"
            "1: the customer raises concerns, friction, or hesitation that could point to "
            "future churn - unresolved complaints, doubts about value, frustration with support "
            "- but does not clearly state an intention to leave or reduce usage.\n"
            "0: the summary expresses no concerns or negative signals relevant to churn."
        )
    )
    competitor_mentioned: bool = Field(
        default=False,
        description="Whether the customer mentioned a competitor or alternative product during the call."
    )
    sentiment: Literal["negative", "neutral", "positive"] = Field(
        default="neutral",
        description=(
            "Overall sentiment perceived of the customer or of how the call went."
        )
    )

SYSTEM_PROMPT = """
You will receive a transcript summary of a call between a customer and a
sales representative.

Extract the following signals from the summary:
- whether the customer signalled intent to expand their use of the product
- the level of churn risk the customer showed
- whether a competitor or alternative product was mentioned
- the overall sentiment perceived based on the summary

Base every field only on what is explicitly stated in the summary. Do not
guess or infer anything not clearly present. When there is no clear signal,
return the default value.
"""

class TranscriptParser:

    def __init__(self, concurrency: int = 50):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.agent = Agent(
            model='google:gemini-3.1-flash-lite',
            model_settings={"temperature": 0.0},
            output_type=TranscriptParserOutput,
            instructions=SYSTEM_PROMPT
        )

    async def parse_batch(self, transcripts: dict[str, str]) -> list[dict[str, any]]:
        results = await asyncio.gather(*[
            self.parse(account_id, transcript)
            for account_id, transcript in transcripts.items()
        ])
        return results

    async def parse(self, account_id: str, transcript: str) -> dict[str, any]:
        try:
            async with self.semaphore:
                agent_result = await self.agent.run(transcript_summary=transcript)
                transcript_output = self.postprocess_output(agent_result.output)
                return transcript_output
        except Exception as e:
            print(f"Error processing account {account_id}: {e}")
            return {}

    def postprocess_output(self, output: TranscriptParserOutput) -> dict[str, any]:
        # at least medium churn risk if negative sentiment detected
        if output.sentiment == "negative":
            output.churn_risk_signal = max(output.churn_risk_signal, 1)

        # highest churn risk if competitor mentioned
        if output.competitor_mentioned:
            output.churn_risk_signal = 2

        # and more business logic rules can be added here as needed
        # ...

        return {
            "gen_expansion_intention": output.expansion_intention,
            "gen_churn_risk_signal": output.churn_risk_signal,
            "gen_competitor_mentioned": output.competitor_mentioned,
            "gen_sentiment": output.sentiment
        }
