from __future__ import annotations

ERROR_TAXONOMY = {
    "R1": "Gold document was not retrieved",
    "R2": "Correct document but wrong snippet",
    "CTX1": "Evidence removed by the context budget",
    "CTX2": "Noisy or contradictory context",
    "G1": "Unsupported generated claim",
    "G2": "Incomplete answer",
    "G3": "Wrong question-type output format",
    "CIT1": "Citation does not support its claim",
    "CIT2": "Invalid PMID citation",
    "S1": "Unsafe or overconfident language",
    "SYS1": "Timeout, parse or provider failure",
}


def validate_error_code(code: str) -> str:
    if code not in ERROR_TAXONOMY:
        raise ValueError(f"Unknown error code: {code}")
    return code
