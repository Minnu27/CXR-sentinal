"""
CXR Sentinel — Phase 3b: claim verification.

Checks every drafted claim (from either report_draft.py path) against the
structured findings it should have come from. This is the anti-hallucination
layer: a claim only survives if there's an actual model number backing it.

Even claims from `draft_report_templated()` are checked here, not skipped —
if someone later swaps in a real LLM path, or if report_draft.py's logic
changes, verification still catches drift instead of trusting it by
construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.report_draft import Claim

POSITIVE_THRESHOLD = 0.5
STATUS_CLAIM_WORDS = {
    "new": ["new"],
    "resolved": ["resolved"],
    "worsening": ["worse", "worsening"],
    "improving": ["improved", "improving"],
}


@dataclass
class VerificationResult:
    finding: str
    claim_text: str
    verdict: str  # SUPPORTED / UNSUPPORTED / INCONSISTENT_STATUS
    reason: str


def verify_claim(claim: Claim, findings_by_name: dict) -> VerificationResult:
    src = findings_by_name.get(claim.finding)

    if src is None:
        return VerificationResult(
            claim.finding, claim.claim_text, "UNSUPPORTED",
            "no matching structured finding exists for this claim at all",
        )

    model_prob = float(src["probability"])
    claims_positive = model_prob >= POSITIVE_THRESHOLD

    # "resolved" claims are legitimately about a currently-low probability
    # (the finding going away), so don't require model_prob to be high for those.
    if claim.source_status != "resolved" and not claims_positive and claim.source_status != "unchanged":
        return VerificationResult(
            claim.finding, claim.claim_text, "UNSUPPORTED",
            f"claim asserts {claim.finding} but model probability is only {model_prob:.2f}, below the {POSITIVE_THRESHOLD} threshold",
        )

    src_status = src.get("status")
    if claim.source_status and claim.source_status != src_status:
        return VerificationResult(
            claim.finding, claim.claim_text, "INCONSISTENT_STATUS",
            f"claim says status='{claim.source_status}' but structured finding says status='{src_status}'",
        )

    for status_key, words in STATUS_CLAIM_WORDS.items():
        if any(w in claim.claim_text.lower() for w in words) and src_status != status_key:
            return VerificationResult(
                claim.finding, claim.claim_text, "INCONSISTENT_STATUS",
                f"claim text implies '{status_key}' but structured status is '{src_status}'",
            )

    return VerificationResult(claim.finding, claim.claim_text, "SUPPORTED", f"model probability {model_prob:.2f} backs this claim")


def verify_claims(claims: list[Claim], findings: list[dict]) -> list[VerificationResult]:
    findings_by_name = {f["finding"]: f for f in findings}
    return [verify_claim(c, findings_by_name) for c in claims]


def results_to_dicts(results: list[VerificationResult]) -> list[dict]:
    return [asdict(r) for r in results]
