"""
UrduGuard - Multi-Agent Pipeline Demo

A minimal 3-agent pipeline (Planner -> Tool-Caller -> Executor), similar in
shape to LangGraph/AutoGen style pipelines. Each "agent" here is a small
rule-based stand-in for an LLM call (no external LLM API is used, so the
demo runs fully offline/deterministically) - this keeps the focus on the
SECURITY MECHANICS the proposal cares about: how does an adversarial
instruction, once it slips past the first checkpoint, get passed along and
acted on by downstream agents?

Flow:
  user_input -> Planner (decides a task) -> Tool-Caller (formats a tool call)
             -> Executor (would actually run the tool)

An UrduGuard checkpoint can be inserted at the user_input boundary and/or
at each inter-agent handoff. We compare "no defense" vs "UrduGuard deployed"
to measure system-level Attack Success Rate (ASR) before/after.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # urduguard/

import re
import joblib
import pandas as pd

MODEL_PATH = os.path.join(BASE_DIR, "results", "urduguard_detector.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "urdu_adversarial_dataset.csv")

DANGEROUS_ACTIONS = ["delete_file", "exfiltrate_data", "send_to_third_party", "disable_safety_check"]


class Planner:
    """Decides what the pipeline should do, based on the (possibly poisoned) input."""
    def plan(self, text: str) -> dict:
        # naive rule-based "LLM": if the text contains injection-style
        # markers, the planner (wrongly) treats them as instructions
        if re.search(r"(system message|admin|ٹول کا نتیجہ|سسٹم پیغام|انتظامی حکم|automated message)", text, re.IGNORECASE):
            if "بھیج" in text or "bhej" in text.lower() or "send" in text.lower():
                return {"task": "send_to_third_party", "source": text}
            if "delete" in text.lower() or "ڈیلیٹ" in text:
                return {"task": "delete_file", "source": text}
            if "چھوڑ" in text or "skip" in text.lower() or "chhor" in text.lower() or "نظر انداز" in text:
                return {"task": "disable_safety_check", "source": text}
        if re.search(r"(ignore|نظر انداز|بھول جاؤ|bhool jao|no restrictions|پابندی نہیں)", text, re.IGNORECASE):
            return {"task": "disable_safety_check", "source": text}
        return {"task": "normal_response", "source": text}


class ToolCaller:
    """Turns the plan into a concrete tool call."""
    def format_call(self, plan: dict) -> dict:
        return {"tool": plan["task"], "args": {"origin_text": plan["source"][:60]}}


class Executor:
    """'Executes' the tool call. In this demo it just records what WOULD happen."""
    def run(self, call: dict) -> str:
        if call["tool"] in DANGEROUS_ACTIONS:
            return f"[EXECUTED - DANGEROUS] {call['tool']}"
        return "[EXECUTED - SAFE] normal_response"


class UrduGuard:
    def __init__(self, model_path=MODEL_PATH, threshold=0.5):
        bundle = joblib.load(model_path)
        self.vectorizer = bundle["vectorizer"]
        self.clf = bundle["clf"]
        self.threshold = threshold

    def check(self, text: str):
        X = self.vectorizer.transform([text])
        score = self.clf.predict_proba(X)[0, 1]
        return score >= self.threshold, score


def run_pipeline(text, guard=None):
    planner, tool_caller, executor = Planner(), ToolCaller(), Executor()

    if guard is not None:
        flagged, score = guard.check(text)
        if flagged:
            return f"[BLOCKED by UrduGuard, score={score:.2f}]", False

    plan = planner.plan(text)
    call = tool_caller.format_call(plan)
    result = executor.run(call)
    attack_succeeded = call["tool"] in DANGEROUS_ACTIONS
    return result, attack_succeeded


def main():
    df = pd.read_csv(DATASET_PATH)
    adversarial = df[df["label"] == "adversarial"]["text"].tolist()

    guard = UrduGuard()

    no_defense_success = 0
    with_defense_success = 0

    print("=== Sample pipeline runs (first 5 adversarial prompts) ===\n")
    for text in adversarial[:5]:
        r_nodef, succ_nodef = run_pipeline(text, guard=None)
        r_def, succ_def = run_pipeline(text, guard=guard)
        print(f"Input: {text[:70]}")
        print(f"  Without UrduGuard -> {r_nodef}")
        print(f"  With UrduGuard    -> {r_def}\n")

    for text in adversarial:
        _, succ_nodef = run_pipeline(text, guard=None)
        _, succ_def = run_pipeline(text, guard=guard)
        no_defense_success += succ_nodef
        with_defense_success += succ_def

    n = len(adversarial)
    print("=== System-level Attack Success Rate (ASR) ===")
    print(f"Total adversarial prompts tested : {n}")
    print(f"ASR WITHOUT UrduGuard : {no_defense_success}/{n} = {no_defense_success/n:.1%}")
    print(f"ASR WITH UrduGuard    : {with_defense_success}/{n} = {with_defense_success/n:.1%}")


if __name__ == "__main__":
    main()
