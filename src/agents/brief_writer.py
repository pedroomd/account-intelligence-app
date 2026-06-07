import json
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class MeetingBrief(BaseModel):
    situation: str = Field(
        description=(
            "One sentence. What kind of account is this right now (in risk of churning, "
            "expanding, mixed signals, steady) and the single most important "
            "reason. No scores or metrics, just the plain-language takeaway."
        )
    )
    what_changed: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Bullets translating the key signals into meaning. Each bullet "
            "must reference a specific number from the provided signals (e.g. "
            "'usage dropped to 38% of seats and they have 12 open tickets')."
            "Leave empty if no key signals present."
        ),
    )
    recommended_focus: str = Field(
        description=(
            "One sentence. Save play, expansion play, or check-in. Be concrete "
            "about what the AE is trying to accomplish on this call."
        )
    )
    talking_points: list[str] = Field(
        max_length=3,
        description=(
            "Specific opening questions or topics. Each one must be "
            "grounded in something from the transcript or the signals - not "
            "generic discovery questions."
        ),
    )
    careful_about: str = Field(
        description=(
            "One sentence. A trap to avoid on this call - e.g. 'they raised "
            "pricing frustration last call, don't lead with the upsell'. If "
            "nothing in the context suggests a trap, say so plainly."
        )
    )


SYSTEM_PROMPT = """
You are a sales analyst helping an Account Executive prepare for a meeting.
You will be given a JSON object describing the customer account: who they are, a status label
derived from validated risk/expansion scores, pre-translated signals (already
in plain language with raw numbers), and the verbatim summary of the most
recent call that sales had with the customer.

Your task is to produce a pre-meeting brief the AE can read in 30 seconds before dialing.

Rules:
- Ground every claim in the provided signals or transcript. Do not invent
  facts, customers, names, or product features that are not in the context.
- Do not restate the raw scores. Translate them.
- No generic sales advice ("build rapport", "understand their needs").
  Every sentence must be specific to this account.
- Talking points must reference something concrete - a number from the
  signals or a phrase/topic from the transcript.
- If the transcript and the metrics tell different stories (e.g. metrics
  flag risk but the call sounded positive), call that mismatch out -
  that's exactly the kind of nuance the AE needs."""


class BriefWriter:

    def __init__(self):
        self.agent = Agent(
            model="google:gemini-3.1-flash-lite",
            model_settings={"temperature": 0.2},
            output_type=MeetingBrief,
            instructions=SYSTEM_PROMPT,
        )

    def generate_brief(self, context: dict) -> MeetingBrief:
        result = self.agent.run_sync(json.dumps(context, indent=2))
        return result.output
