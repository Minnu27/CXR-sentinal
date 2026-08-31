"""
CXR Sentinel — Phase 3a: report drafting.

Two real paths, pick one:

1. `draft_report_templated()` — deterministic, runs today, zero setup, zero
   API cost. Every sentence is built directly from your model's actual
   numbers. This is NOT a placeholder — it's a legitimate, defensible way to
   turn structured findings into report language, and it's easier to verify
   than an LLM's output because it can't say anything the numbers didn't
   already say.

2. `draft_report_llm()` — a real integration point for an actual LLM call
   (Anthropic API shown; swap for whichever you have a key for), used only
   to improve phrasing quality, NOT to add new claims. The prompt explicitly
   instructs the model to only rephrase what's given, output strict JSON,
   and never introduce a finding that isn't in the input. This function is
   not wired to run automatically — it needs an API key supplied at call
   time. Everything downstream (claim_verify.py) works identically whether
   the claims came from path 1 or path 2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Claim:
    finding: str
    claim_text: str
    location: str
    source_probability: float
    source_status: str | None  # temporal status from Phase 2, if available


PROBABILITY_TIERS = [
    (0.85, "high confidence of"),
    (0.65, "likely"),
    (0.50, "possible"),
]

REGION_HINTS = {
    "cardiomegaly": "cardiac silhouette",
    "pleural_effusion": "costophrenic angle / pleural space",
    "lung_opacity": "lung parenchyma",
}


def _confidence_phrase(prob: float) -> str | None:
    for cutoff, phrase in PROBABILITY_TIERS:
        if prob >= cutoff:
            return phrase
    return None  # below the lowest tier — not asserted


def draft_report_templated(findings: list[dict]) -> list[Claim]:
    """
    findings: list of dicts, each at minimum {"finding": str, "probability": float},
    optionally {"status": "new"/"worsening"/"improving"/"resolved"/"unchanged"} from Phase 2.

    Returns one Claim per finding that clears the lowest confidence tier.
    Findings below the lowest tier are omitted entirely (silence, not a
    fabricated "no evidence of X" for every possible finding on every image).
    """
    claims = []
    for f in findings:
        name = f["finding"]
        prob = float(f["probability"])
        status = f.get("status")
        region = REGION_HINTS.get(name, "affected region")
        display_name = name.replace("_", " ")

        if status == "new":
            text = f"New {display_name} is noted, {region}."
        elif status == "resolved":
            text = f"Previously noted {display_name} has resolved."
        elif status in ("worsening",) or (status and status.startswith("worsening")):
            text = f"{display_name.capitalize()} is present and appears worse compared with the prior study."
        elif status in ("improving",) or (status and status.startswith("improving")):
            text = f"{display_name.capitalize()} is present but appears improved compared with the prior study."
        else:
            phrase = _confidence_phrase(prob)
            if phrase is None:
                continue  # not confident enough to assert anything — omit, don't guess
            text = f"There is {phrase} {display_name}, {region}."

        claims.append(
            Claim(finding=name, claim_text=text, location=region, source_probability=prob, source_status=status)
        )
    return claims


LLM_SYSTEM_PROMPT = """You are drafting radiology report language from pre-computed model findings.
Rules:
- Only rephrase the findings given to you. Never introduce a finding, location, or severity that isn't in the input.
- Every claim you output must be traceable to exactly one input finding.
- Output strict JSON: a list of objects with keys finding, claim_text, location.
- No prose outside the JSON.
"""


def draft_report_llm(findings: list[dict], api_key: str, model: str = "claude-sonnet-4-6") -> list[Claim]:
    """
    Real integration point — requires `pip install anthropic` and a real API
    key passed in at call time. Not called by default anywhere in this repo.
    Improves phrasing only; claim_verify.py still checks every output claim
    against the same structured findings regardless of which path produced it.
    """
    import anthropic  # local import: only required if you actually use this path

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = f"Findings:\n{json.dumps(findings, indent=2)}\n\nDraft the report claims as JSON."

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=LLM_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw_text = response.content[0].text
    parsed = json.loads(raw_text)

    claims = []
    findings_by_name = {f["finding"]: f for f in findings}
    for item in parsed:
        src = findings_by_name.get(item["finding"], {})
        claims.append(
            Claim(
                finding=item["finding"],
                claim_text=item["claim_text"],
                location=item.get("location", "unknown"),
                source_probability=float(src.get("probability", 0.0)),
                source_status=src.get("status"),
            )
        )
    return claims


def claims_to_json(claims: list[Claim]) -> str:
    return json.dumps([asdict(c) for c in claims], indent=2)
