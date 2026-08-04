"""
UrduGuard - Baseline "English-centric" guardrail

Simulates how most current open-source guardrail keyword/pattern lists work:
they are built from English jailbreak/injection phrases (e.g. "ignore previous
instructions", "you are DAN", "no restrictions", "system message:", "act as").

This baseline is intentionally simple (keyword + regex matching, the same
mechanism many lightweight guardrails use before an ML classifier stage) so
we can measure how much it misses when the SAME attack intent is expressed
in Urdu script or Roman Urdu instead of English.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # urduguard/

import re
import pandas as pd

ENGLISH_PATTERNS = [
    r"ignore (all|previous|prior) instructions",
    r"you are (now )?dan",
    r"do anything now",
    r"no (restrictions|rules|filter)",
    r"act as (an? )?(unrestricted|jailbroken|free) ai",
    r"system message",
    r"admin (command|override)",
    r"debug mode",
    r"pretend (you|to) (are|be)",
    r"disregard (your|the) (rules|training|guidelines)",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in ENGLISH_PATTERNS]


def baseline_flag(text: str) -> bool:
    """Return True if the English-pattern guardrail would flag this text."""
    return any(p.search(text) for p in _compiled)


def run(dataset_path: str):
    df = pd.read_csv(dataset_path)
    df["baseline_flagged"] = df["text"].apply(baseline_flag)

    adversarial = df[df["label"] == "adversarial"]
    caught = adversarial["baseline_flagged"].sum()
    total = len(adversarial)
    miss_rate = 1 - (caught / total) if total else 0

    # false positives on benign traffic
    benign = df[df["label"] == "benign"]
    fp = benign["baseline_flagged"].sum()

    print("=== Baseline English-keyword guardrail ===")
    print(f"Adversarial prompts total       : {total}")
    print(f"Adversarial prompts CAUGHT       : {caught}")
    print(f"Adversarial prompts MISSED       : {total - caught}")
    print(f"Miss rate (false negative rate)  : {miss_rate:.1%}")
    print(f"False positives on benign traffic: {fp} / {len(benign)}")

    # breakdown by script (urdu_script vs roman_urdu vs any english leakage)
    by_script = adversarial.groupby("script")["baseline_flagged"].agg(["sum", "count"])
    by_script["miss_rate"] = 1 - by_script["sum"] / by_script["count"]
    print("\nBreakdown by script:")
    print(by_script)

    return df


if __name__ == "__main__":
    run(os.path.join(BASE_DIR, "dataset", "urdu_adversarial_dataset.csv"))
