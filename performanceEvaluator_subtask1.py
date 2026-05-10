"""
performanceEvaluator_subtask1.py
---------------------------------
Run from: Case Study/
Command:  python performanceEvaluator_subtask1.py

Option 1 - Baseline evaluation:
  For each language:
    1. Create my_baselines/<LANG> folder
    2. Generate majority and random baselines
    3. Remove empty lines and BOM from generated files
    4. Run scorer and print results

Option 2 - Personal method evaluation:
  Reads your prediction file from:
    personal_results/subtask_1/NLP_prediction_Subtask1.txt
  Runs the scorer against the gold file for the detected languages.

Option 3 - Performance comparison:
  Compares personal method vs majority baseline vs random baseline,
  printed side by side for each detected language.
"""

import os
import sys
import subprocess


# - path configuration -

BASELINE_SCRIPT = os.path.join("baselines", "semeval2025task10-scorers-baselines-v3", "subtask1_baseline.py")
SCORER_SCRIPT   = os.path.join("baselines", "semeval2025task10-scorers-baselines-v3", "subtask1_scorer.py")

LANGUAGES = ["EN", "BG", "HI", "PT", "RU"]

PERSONAL_PRED_FILE = os.path.join("personal_results", "subtask_1", "NLP_prediction_Subtask1.txt")


def train_file(lang):
    ''' Returns the path to the training file for the given language. '''
    return os.path.join("training_NLP", "training", lang, "subtask-1-annotations.txt")

def dev_file(lang):
    ''' Returns the path to the development file for the given language. '''
    return os.path.join("test_NLP", "test", lang, "subtask-1-entity-mentions.txt")

def gold_file(lang):
    ''' Returns the path to the gold file for the given language. '''
    return os.path.join("test_NLP", "test", lang, "subtask-1-annotations.txt")

def output_dir(lang):
    ''' Returns the path to the output directory for the given language. '''
    return os.path.join("my_baselines", lang)


# - helper functions -

def run(cmd):
    ''' Runs a command using subprocess and prints the output. '''
    print(f"\n> {' '.join(cmd)}")
    # command execution with output capture
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # print stdout if present
    if result.stdout.strip():
        print(result.stdout.strip())
        
    # print stderr if present
    if result.stderr.strip():
        print("[STDERR]", result.stderr.strip())
    return result.returncode

def fix_file(path):
    ''' Removes BOM and empty lines from the given file. '''
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return
    
    # read files removing BOM
    with open(path, "r", encoding="utf-8-sig") as f:
        # keep only non-empty lines
        lines = [l for l in f.readlines() if l.strip()]
    # rewrite file with cleaned lines
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    print(f"  Fixed (BOM + empty lines removed): {path}")

def run_scorer(gold, pred):
    result = subprocess.run(
        [sys.executable, SCORER_SCRIPT, "-g", gold, "-p", pred],
        capture_output=True, text=True
    )
    return result.stdout.strip() or result.stderr.strip()

def parse_scores(line):
    ''' Parses a line of scores into its components. '''
    return line.split("\t") if "\t" in line else line.split()

def print_summary(results, label):
    ''' Prints a summary table of results. '''
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY ({label})")
    print(f"  {'LANG':<6} {'EMR':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Acc'}")
    print(f"  {'-'*54}")
    # print results for each language
    for lang, line in results.items():
        # check metric presence and parse accordingly
        cols = parse_scores(line)
        if len(cols) >= 5:
            print(f"  {lang:<6} {cols[0]:<8} {cols[1]:<8} {cols[2]:<8} {cols[3]:<8} {cols[4]}")
        else:
            print(f"  {lang:<6} {line}")
    print("=" * 60)

def ensure_baselines_exist(lang):
    ''' Checks if baseline files exist for the given language, and generates them if not. '''
    maj_path = os.path.join(output_dir(lang), "baseline_majority.txt")
    rnd_path = os.path.join(output_dir(lang), "baseline_random.txt")

    if not os.path.exists(maj_path) or not os.path.exists(rnd_path):
        print(f"\n  Baselines not found for {lang}, generating them now...")
        os.makedirs(output_dir(lang), exist_ok=True)

        run([sys.executable, BASELINE_SCRIPT,
             "--train_file", train_file(lang),
             "--dev_file",   dev_file(lang),
             "--output_dir", output_dir(lang),
             "--baseline_type", "majority"])

        run([sys.executable, BASELINE_SCRIPT,
             "--dev_file",   dev_file(lang),
             "--output_dir", output_dir(lang),
             "--baseline_type", "random"])

        fix_file(maj_path)
        fix_file(rnd_path)
    else:
        print(f"  Baselines found for {lang}.")

def detect_languages(filepath):
    ''' Detects which languages are present in the given prediction file based on filename patterns. '''
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    detected = [lang for lang in LANGUAGES if f"_{lang}_" in content or content.startswith(lang)]
    return detected if detected else ["EN"]


# - option 1: baseline evaluation -

def run_baseline_evaluation():
    print("\n" + "=" * 60)
    print("  SUBTASK 1 - Baseline generation + scoring")
    print("=" * 60)

    results_majority = {}
    results_random = {}

    for lang in LANGUAGES:
        print(f"\n  LANGUAGE: {lang}")

        os.makedirs(output_dir(lang), exist_ok=True)
        print(f"  Folder created: {output_dir(lang)}")

        run([sys.executable, BASELINE_SCRIPT,
             "--train_file", train_file(lang),
             "--dev_file",   dev_file(lang),
             "--output_dir", output_dir(lang),
             "--baseline_type", "majority"])

        run([sys.executable, BASELINE_SCRIPT,
             "--dev_file",   dev_file(lang),
             "--output_dir", output_dir(lang),
             "--baseline_type", "random"])

        fix_file(os.path.join(output_dir(lang), "baseline_majority.txt"))
        fix_file(os.path.join(output_dir(lang), "baseline_random.txt"))

        print(f"\n  Scorer - {lang} majority:")
        score_maj = run_scorer(gold_file(lang), os.path.join(output_dir(lang), "baseline_majority.txt"))
        print(f"  {score_maj}")
        results_majority[lang] = score_maj

        print(f"\n  Scorer - {lang} random:")
        score_rnd = run_scorer(gold_file(lang), os.path.join(output_dir(lang), "baseline_random.txt"))
        print(f"  {score_rnd}")
        results_random[lang] = score_rnd

    print_summary(results_majority, "majority baseline")
    print_summary(results_random, "random baseline")


# - option 2: personal method evaluation -

def run_personal_evaluation():
    print("\n" + "=" * 60)
    print("  SUBTASK 1 - Personal method scoring")
    print("=" * 60)

    if not os.path.exists(PERSONAL_PRED_FILE):
        print(f"\n  ERROR: Prediction file not found at:")
        print(f"  {PERSONAL_PRED_FILE}")
        print(f"\n  Please place your prediction file at that path and try again.")
        return

    print(f"\n  Prediction file found: {PERSONAL_PRED_FILE}")
    fix_file(PERSONAL_PRED_FILE)

    detected_langs = detect_languages(PERSONAL_PRED_FILE)
    print(f"  Languages detected in prediction file: {', '.join(detected_langs)}")

    results = {}

    for lang in detected_langs:
        gold = gold_file(lang)
        if not os.path.exists(gold):
            print(f"\n  Gold file not found for {lang}: {gold} — skipping.")
            continue

        print(f"\n  Scorer - {lang}:")
        score_line = run_scorer(gold, PERSONAL_PRED_FILE)
        print(f"  {score_line}")
        results[lang] = score_line

    if results:
        print_summary(results, "personal method")


# - option 3: performance comparison -

def run_comparison():
    print("\n" + "=" * 60)
    print("  SUBTASK 1 - Performance comparison")
    print("=" * 60)

    if not os.path.exists(PERSONAL_PRED_FILE):
        print(f"\n  ERROR: Personal prediction file not found at:")
        print(f"  {PERSONAL_PRED_FILE}")
        print(f"\n  Please place your prediction file at that path and try again.")
        return

    fix_file(PERSONAL_PRED_FILE)

    detected_langs = detect_languages(PERSONAL_PRED_FILE)
    print(f"  Languages detected in prediction file: {', '.join(detected_langs)}")

    for lang in detected_langs:
        gold = gold_file(lang)
        if not os.path.exists(gold):
            print(f"\n  Gold file not found for {lang} — skipping.")
            continue

        ensure_baselines_exist(lang)

        score_personal = run_scorer(gold, PERSONAL_PRED_FILE)
        score_majority = run_scorer(gold, os.path.join(output_dir(lang), "baseline_majority.txt"))
        score_random   = run_scorer(gold, os.path.join(output_dir(lang), "baseline_random.txt"))

        print(f"\n{'='*60}")
        print(f"  COMPARISON - {lang}")
        print(f"  {'METHOD':<22} {'EMR':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Acc'}")
        print(f"  {'-'*54}")

        # print results for personal method and baselines side by side
        for label, score_line in [
            ("Personal method", score_personal),
            ("Majority baseline", score_majority),
            ("Random baseline", score_random),
        ]:
            cols = parse_scores(score_line)
            if len(cols) >= 5:
                print(f"  {label:<22} {cols[0]:<8} {cols[1]:<8} {cols[2]:<8} {cols[3]:<8} {cols[4]}")
            else:
                print(f"  {label:<22} {score_line}")

    print("=" * 60)


# - main -

def main():
    print("=" * 60)
    print("  SUBTASK 1 - Performance Evaluator")
    print("=" * 60)
    print()
    print("  Select evaluation mode:")
    print("  1 - Evaluate baseline performance (majority + random)")
    print("  2 - Evaluate your own NLP method")
    print("  3 - Compare personal method vs baselines")
    print()

    choice = input("  Enter 1, 2 or 3: ").strip()

    if choice == "1":
        run_baseline_evaluation()
    elif choice == "2":
        run_personal_evaluation()
    elif choice == "3":
        run_comparison()
    else:
        print("  Invalid choice. Please enter 1, 2 or 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()