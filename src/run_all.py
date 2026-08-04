"""
UrduGuard - Run full pipeline end-to-end and save a consolidated summary.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # urduguard/
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline_guardrail
import multi_agent_pipeline as pipe

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "urdu_adversarial_dataset.csv")

def main():
    # 1. Baseline
    df = baseline_guardrail.run(DATASET_PATH)
    adversarial = df[df["label"] == "adversarial"]
    baseline_catch_rate = adversarial["baseline_flagged"].mean()

    # 2. Detector metrics (already saved by train_detector.py)
    with open(os.path.join(BASE_DIR, "results", "detector_metrics.json")) as f:
        detector_metrics = json.load(f)

    # 3. Multi-agent ASR
    guard = pipe.UrduGuard()
    adv_texts = adversarial["text"].tolist()
    no_def = sum(pipe.run_pipeline(t, guard=None)[1] for t in adv_texts)
    with_def = sum(pipe.run_pipeline(t, guard=guard)[1] for t in adv_texts)
    n = len(adv_texts)

    summary = {
        "baseline_english_guardrail_catch_rate": round(float(baseline_catch_rate), 4),
        "urduguard_detector_test_metrics": detector_metrics,
        "system_level_ASR_without_urduguard": round(no_def / n, 4),
        "system_level_ASR_with_urduguard": round(with_def / n, 4),
        "n_adversarial_prompts": n,
    }

    with open(os.path.join(BASE_DIR, "results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # 4. Chart: detection/catch rate comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(
        ["English-keyword\nbaseline", "UrduGuard\ndetector (recall)"],
        [baseline_catch_rate * 100, detector_metrics["recall"] * 100],
        color=["#c0392b", "#1f8a4c"],
    )
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Adversarial prompts caught (%)")
    axes[0].set_title("Detection rate: baseline vs UrduGuard")
    for i, v in enumerate([baseline_catch_rate * 100, detector_metrics["recall"] * 100]):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    axes[1].bar(
        ["Without\nUrduGuard", "With\nUrduGuard"],
        [no_def / n * 100, with_def / n * 100],
        color=["#c0392b", "#1f8a4c"],
    )
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Attack Success Rate (%)")
    axes[1].set_title("System-level ASR in multi-agent pipeline")
    for i, v in enumerate([no_def / n * 100, with_def / n * 100]):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "results", "results_chart.png"), dpi=150)
    print("\nSaved chart ->", os.path.join(BASE_DIR, "results", "results_chart.png"))


if __name__ == "__main__":
    main()
