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

PERSONAL_DIR = "personal_results/subtask_2/"

# All personal methods: (display label, filename)
PERSONAL_METHODS = [
    ("Linear SVM",         "predictions_subtask2_EN_LinearSVM.txt"),
    ("SVM Regularization", "predictions_subtask2_EN_Regularization.txt"),
    ("DistilBERT",         "predictions_subtask2_EN_bert.txt"),
]

LANG_PREFIX = "EN_"


# Helpers

def load_classes(filepath):
    """Load labels from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line for line in f.read().split('\n') if line.strip()]


def load_en_ids(gold_file):
    """Extract ONLY English IDs from gold file."""
    ids = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(LANG_PREFIX):
                doc_id = line.split('\t')[0]
                ids.append(doc_id)
    return ids


def generate_official_baseline(classes_coarse, classes_fine, ids):
    """Generate predictions using the OFFICIAL baseline."""
    dummy_texts = ["text"] * len(ids)
    baseline = T10ST2RandomBaseline(classes_coarse, classes_fine)
    os.makedirs(os.path.dirname(BASELINE_OUTPUT), exist_ok=True)
    baseline.predict_and_write(dummy_texts, ids, BASELINE_OUTPUT)
    print(f"\n  Baseline predictions written to:")
    print(f"  {BASELINE_OUTPUT}")


def evaluate_file(prediction_file):
    """Evaluate predictions using OFFICIAL scorer."""
    return read_and_evaluate(
        prediction_file,
        GOLD_FILE,
        NARRATIVES_FILE,
        SUBNARRATIVES_FILE
    )


def print_table_header():
    print(f"\n  {'Method':<30} {'F1@coarse':>14} {'F1@fine':>14}")
    print(f"  {'-' * 60}")


def print_table_row(label, f1_c, sd_c, f1_f, sd_f):
    print(
        f"  {label:<30} "
        f"{f1_c:.3f} ({sd_c:.3f})  "
        f"{f1_f:.3f} ({sd_f:.3f})"
    )


# Option 1 - Baseline
def option1(classes_coarse, classes_fine):
    print("\n Option 1: Random Baseline ")
    ids = load_en_ids(GOLD_FILE)
    generate_official_baseline(classes_coarse, classes_fine, ids)
    f1_c, sd_c, f1_f, sd_f = evaluate_file(BASELINE_OUTPUT)
    print_table_header()
    print_table_row("Random Baseline (EN)", f1_c, sd_c, f1_f, sd_f)


# Option 2 - All personal methods
def option2():
    print("\n Option 2: Personal Methods ")
    print_table_header()

    results = {}
    for label, filename in PERSONAL_METHODS:
        filepath = os.path.join(PERSONAL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [!] File not found: {filepath}")
            continue
        f1_c, sd_c, f1_f, sd_f = evaluate_file(filepath)
        print_table_row(f"{label} (EN)", f1_c, sd_c, f1_f, sd_f)
        results[label] = (f1_c, sd_c, f1_f, sd_f)

    return results


# Option 3 - Comparison: baseline vs all personal methods
def option3(classes_coarse, classes_fine):
    print("\n Option 3: Comparison — Baseline vs Personal Methods ")

    # Baseline
    ids = load_en_ids(GOLD_FILE)
    generate_official_baseline(classes_coarse, classes_fine, ids)
    b_f1_c, b_sd_c, b_f1_f, b_sd_f = evaluate_file(BASELINE_OUTPUT)

    print_table_header()
    print_table_row("Random Baseline (EN)", b_f1_c, b_sd_c, b_f1_f, b_sd_f)
    print(f"  {'-' * 60}")

    for label, filename in PERSONAL_METHODS:
        filepath = os.path.join(PERSONAL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [!] File not found: {filepath}")
            continue

        p_f1_c, p_sd_c, p_f1_f, p_sd_f = evaluate_file(filepath)
        print_table_row(f"{label} (EN)", p_f1_c, p_sd_c, p_f1_f, p_sd_f)



# Main menu
def main():
    print("=" * 60)
    print("  SemEval 2025 Task 10: Subtask 2 Evaluator")
    print("=" * 60)

    print("\n  1: Random baseline evaluation")
    print("  2: Personal methods evaluation ")
    print("  3: Performance comparison (baseline vs all personal methods)")
    print("  q: Quit")

    choice = input("\n  Choose an option: ").strip().lower()

    if choice == 'q':
        sys.exit(0)

    if choice not in ('1', '2', '3'):
        print("  Invalid choice.")
        sys.exit(1)

    classes_coarse = load_classes(NARRATIVES_FILE)
    classes_fine   = load_classes(SUBNARRATIVES_FILE)

    if choice == '1':
        option1(classes_coarse, classes_fine)
    elif choice == '2':
        option2()
    elif choice == '3':
        option3(classes_coarse, classes_fine)

    print()


if __name__ == '__main__':
    main()