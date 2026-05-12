"""
performanceEvaluator_subtask2.py
---------------------------------
Unified evaluator for SemEval 2025 Task 10 - Subtask 2.

This version directly uses:
- subtask2_baseline_random.py
- subtask2_scorer.py

so the behavior perfectly matches the official implementation.
"""

import os
import sys

# Import official baseline + scorer

from subtask2_resolution.subtask2_baseline_random import T10ST2RandomBaseline
from subtask2_resolution.subtask2_scorer import read_and_evaluate


# Paths

GOLD_FILE = "subtask2_resolution/subtask-2-annotations.txt"

NARRATIVES_FILE = (
    "subtask2_resolution/subtask2_narratives.txt"
)

SUBNARRATIVES_FILE = (
    "subtask2_resolution/subtask2_subnarratives.txt"
)

BASELINE_OUTPUT = (
    "subtask2_resolution/subtask2_baselines/"
    "test_baselines/st2_en_random_predictions.txt"
)

PERSONAL_OUTPUT = (
    "personal_results/subtask_2/"
    "NLP_prediction_Subtask2.txt"
)

LANG_PREFIX = "EN_"


# Helpers

def load_classes(filepath):
    """
    Load labels from a text file.
    """

    with open(filepath, 'r', encoding='utf-8') as f:
        return [line for line in f.read().split('\n') if line.strip()]


def load_en_ids(gold_file):
    """
    Extract ONLY English IDs from gold file.
    """

    ids = []

    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(LANG_PREFIX):
                doc_id = line.split('\t')[0]
                ids.append(doc_id)

    return ids


def generate_official_baseline(classes_coarse,
                               classes_fine,
                               ids):
    """
    Generate predictions using the OFFICIAL baseline.
    """

    # Dummy texts required by official baseline
    dummy_texts = ["text"] * len(ids)

    baseline = T10ST2RandomBaseline(
        classes_coarse,
        classes_fine
    )

    os.makedirs(
        os.path.dirname(BASELINE_OUTPUT),
        exist_ok=True
    )

    baseline.predict_and_write(
        dummy_texts,
        ids,
        BASELINE_OUTPUT
    )

    print(f"\n  Baseline predictions written to:")
    print(f"  {BASELINE_OUTPUT}")


def evaluate_file(prediction_file):
    """
    Evaluate predictions using OFFICIAL scorer.
    """

    return read_and_evaluate(
        prediction_file,
        GOLD_FILE,
        NARRATIVES_FILE,
        SUBNARRATIVES_FILE
    )


def print_results(label, f1_c, sd_c, f1_f, sd_f):
    print(f"\n  {'Method':<30} {'F1@coarse':>14} {'F1@fine':>14}")
    print(f"  {'-' * 60}")
    print(
        f"  {label:<30} "
        f"{f1_c:.3f} ({sd_c:.3f})  "
        f"{f1_f:.3f} ({sd_f:.3f})"
    )


# Option 1 - Baseline
def option1(classes_coarse, classes_fine):
    print("\n Option 1: Random Baseline ")
    ids = load_en_ids(GOLD_FILE)

    generate_official_baseline(
        classes_coarse,
        classes_fine,
        ids
    )

    f1_c, sd_c, f1_f, sd_f = evaluate_file(BASELINE_OUTPUT)

    print_results("Random Baseline (EN)",f1_c, sd_c, f1_f, sd_f)


# Option 2 - Personal method
def option2():
    print("\n Option 2: Personal Method ")

    if not os.path.exists(PERSONAL_OUTPUT):
        print(f"  [!] Prediction file not found:")
        print(f"      {PERSONAL_OUTPUT}")
        return None

    f1_c, sd_c, f1_f, sd_f = evaluate_file(
        PERSONAL_OUTPUT
    )

    print_results( "Personal Method (EN)", f1_c, sd_c, f1_f, sd_f)
    return f1_c, sd_c, f1_f, sd_f


# Option 3 - Comparison
def option3(classes_coarse, classes_fine):
    print("\n Option 3: Comparison ")

    # Baseline
    ids = load_en_ids(GOLD_FILE)

    generate_official_baseline(
        classes_coarse,
        classes_fine,
        ids
    )

    b_f1_c, b_sd_c, b_f1_f, b_sd_f = evaluate_file(BASELINE_OUTPUT)

    # Personal
    if not os.path.exists(PERSONAL_OUTPUT):
        print(f"  [!] Personal prediction file not found:")
        print(f"      {PERSONAL_OUTPUT}")
        return

    p_f1_c, p_sd_c, p_f1_f, p_sd_f = evaluate_file(PERSONAL_OUTPUT)

    # Comparison table
    print(f"\n  {'Method':<30} {'F1@coarse':>14} {'F1@fine':>14}")
    print(f"  {'-' * 60}")
    print(
        f"  {'Random Baseline (EN)':<30} "
        f"{b_f1_c:.3f} ({b_sd_c:.3f})  "
        f"{b_f1_f:.3f} ({b_sd_f:.3f})"
    )

    print(
        f"  {'Personal Method (EN)':<30} "
        f"{p_f1_c:.3f} ({p_sd_c:.3f})  "
        f"{p_f1_f:.3f} ({p_sd_f:.3f})"
    )

    delta_c = p_f1_c - b_f1_c
    delta_f = p_f1_f - b_f1_f

    sign_c = "+" if delta_c >= 0 else ""
    sign_f = "+" if delta_f >= 0 else ""

    print(
        f"\n  {'Delta (personal - baseline)':<30} "
        f"{sign_c}{delta_c:.3f}{'':>10} "
        f"{sign_f}{delta_f:.3f}"
    )


# Main menu
def main():

    print("=" * 60)
    print("  SemEval 2025 Task 10: Subtask 2 Evaluator")
    print("=" * 60)

    print("\n  1: Random baseline evaluation")
    print("  2: Personal method evaluation")
    print("  3: Performance comparison")
    print("  q: Quit")

    choice = input("\n  Choose an option: ").strip().lower()

    if choice == 'q':
        sys.exit(0)

    if choice not in ('1', '2', '3'):

        print("  Invalid choice.")
        sys.exit(1)

    # Load classes
    classes_coarse = load_classes(NARRATIVES_FILE)

    classes_fine = load_classes(SUBNARRATIVES_FILE)

    # Execute selected option
    if choice == '1':
        option1(classes_coarse, classes_fine)

    elif choice == '2':
        option2()

    elif choice == '3':
        option3(classes_coarse, classes_fine)

    print()


if __name__ == '__main__':
    main()