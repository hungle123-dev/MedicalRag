from __future__ import annotations


def superiority_gate(
    mean_delta: float,
    ci95_low: float,
    failure_rate_delta: float,
    p95_latency_ratio: float,
    minimum_effect: float = 0.01,
) -> dict[str, object]:
    checks = {
        "minimum_effect": mean_delta >= minimum_effect,
        "positive_paired_ci": ci95_low > 0,
        "failure_guard": failure_rate_delta <= 0.005,
        "latency_guard": p95_latency_ratio <= 2.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def noninferiority_gate(
    mean_delta: float, ci95_low: float, margin: float = 0.01
) -> dict[str, object]:
    checks = {"mean_within_margin": mean_delta >= -margin, "ci_within_margin": ci95_low > -margin}
    return {"passed": all(checks.values()), "checks": checks, "margin": margin}
