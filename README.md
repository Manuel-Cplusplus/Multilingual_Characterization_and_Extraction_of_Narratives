# Multilingual Characterization and Extraction of Narratives from Online News

> **SemEval-2025 Task 10** - Subtask 1: Entity Framing · Subtask 2: Narrative Classification
>
> Manuel Carlucci - University of Bari "Aldo Moro" · `m.carlucci69@studenti.uniba.it`

---

## Overview

This repository contains the code, notebooks, and evaluation utilities developed for [SemEval-2025 Task 10](https://propaganda.math.unipd.it/semeval2025task10/), which focuses on the multilingual identification and classification of narratives in online news articles.

Both subtasks are framed as **multi-label, multi-class classification** problems. Given the limited size of the available training data, the core approach relies on classical machine learning pipelines built on pre-trained word embeddings, while also benchmarking transformer-based and large language model alternatives.

### Subtask 1 - Entity Framing

Assigns one or more semantic roles (Protagonist, Antagonist, Innocent) to named entities mentioned in news articles.

**Approach:** Two-stage Logistic Regression pipeline on 600-dimensional spaCy `en_core_web_md` feature vectors (mention + context embeddings), with balanced class weights.


### Subtask 2 - Narrative Classification

Assigns one or more narrative labels from a two-level taxonomy (22 dominant narratives, 91 sub-narratives) to full news articles, covering Climate Change and the Ukraine-Russia War.

**Approaches compared:** TF-IDF weighted Word2Vec + Linear SVM · DistilBERT fine-tuning · LLaMA-3.2-3B with LoRA (via Unsloth).


The SVM baseline outperforms both transformer-based approaches, consistent with the low-data regime: with only 399 training samples across 80+ imbalanced sub-narrative labels, linear classifiers with TF-IDF representations generalise more effectively than larger neural models.

---

## Repository Structure

- performanceEvaluator_subtask1.py - interactive evaluator for Subtask 1. Supports baseline generation, scoring, and side-by-side comparison with score delta.
- performanceEvaluator_subtask2.py - same as above for Subtask 2, English articles only.
- baselines/ - official task scripts and pre-generated baseline outputs (scorers, majority/random generators).
- my_baselines/ - locally regenerated baselines for all supported languages (BG, EN, HI, PT, RU), created to fix identifier mismatches in the official files. Only for subtaask 1
- personal_results/subtask_1/ - predictions from the spaCy + LR pipeline. 
- personal_results/subtask_2/ - prediction files for each method: SVM, DistilBERT, and LLaMA.
- subtask1_resolution/ and subtask2_resolution/ - gold annotations and official scorers for each subtask.
- test_NLP/ and training_NLP/ - test and training data organised by language and subtask.

---

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
python -m venv myenv
myenv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python -m venv myenv
source myenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Evaluators

Both evaluators expose an **interactive menu** to choose between baseline-only evaluation, personal method evaluation, and a side-by-side comparison with score delta.

**Subtask 1:**
```bash
python performanceEvaluator_subtask1.py
```

**Subtask 2:**
```bash
python performanceEvaluator_subtask2.py
```

To reproduce Subtask 1 results without retraining, the pre-trained models saved in `personal_results/subtask_1/models/` can be loaded directly.

---

## Reproducing the Experiments

### Subtask 1

1. Run the Subtask 1 notebook to generate predictions.
2. Run `performanceEvaluator_subtask1.py` pointing to `personal_results/subtask_1/NLP_prediction_Subtask1.txt`.

> **Known issue:** The official baseline prediction files contain document identifiers that do not match the English test set. Baselines were regenerated locally using `baselines/semeval2025task10-scorers-baselines-v3/subtask1_baseline.py`. Additional preprocessing (empty lines, UTF-8 BOM on Windows) is handled by the evaluator script.

### Subtask 2

1. Run the Subtask 2 notebook (SVM / DistilBERT / LLaMA cell of choice).
2. Run `performanceEvaluator_subtask2.py` - it evaluates English articles only and generates the random baseline automatically for comparison.

---

## Supported Languages

`EN` · `BG` · `HI` · `PT` · `RU`

Subtask 1 and 2 evaluation in this repository covers **English only**. Training and evaluation on the remaining languages is left as future work.

---

## Citation

If you use this repository, please cite the shared task paper:

```bibtex
@inproceedings{piskorski-etal-2025-semeval,
  title     = {{SemEval}-2025 Task 10: Multilingual Characterization and Extraction of Narratives from Online News},
  author    = {Piskorski, Jakub and others},
  booktitle = {Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)},
  publisher = {Association for Computational Linguistics},
  year      = {2025},
  url       = {https://aclanthology.org/2025.semeval-1.331/}
}
```

---

## License

This project is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.