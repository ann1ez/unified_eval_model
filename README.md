# Unified Evals: LLM Judge Scoring + Supervised Learning

This document describes the two-notebook pipeline for evaluating LLM output quality using LLM-as-judge scoring followed by supervised learning.

## Pipeline Overview

```
unified_eval_datasets.ipynb          unified_eval_learning.ipynb
─────────────────────────────        ─────────────────────────────
1. Build eval datasets (800 rows)    1. Load scored dataset
2. Define quality criteria (8 per    2. Train 7 classifiers + 2 baselines
   task)                             3. Cross-validate (stratified 5-fold)
3. Score with LLM judge              4. Explainability analysis
4. Export feature CSVs               5. Final test-set evaluation
```

---

## Notebook 1: `unified_eval_datasets.ipynb`

### Purpose

Generates benchmark evaluation datasets and scores them with an LLM judge to produce binary feature vectors for each example.

### Datasets

Three evaluation datasets are built from public benchmarks, each with 800 rows and a unified schema (`input`, `output`, `label`):

| Dataset | Source Benchmark | Input | Output | Label Logic |
|---|---|---|---|---|
| **Translation** | WMT MQM Human Evaluation (`RicardoRei/wmt-mqm-human-evaluation`) | Non-English source text | English MT output | Median split on MQM score within (language pair, year) |
| **Summarization** | FRANK (`frank/data/`) | News article | Model-generated summary | Human factuality annotation (1.0 = pass, else fail) |
| **Extraction** | CoNLL-2003 NER (`conll2003`) | Sentence text | Predicted entity list (JSON) | Exact match between predicted and gold entity sets |

All datasets use stratified sampling to maintain a reasonable label balance. A fixed random seed (229) ensures reproducibility.

### Quality Guideline Specifications

For each task, 8 quality criteria are defined as text-based guidelines. These are used to construct LLM judge scorers that produce binary (yes/no) assessments per criterion.

**Translation criteria**: meaning_preservation, no_added_info, coverage, named_entities_numbers, terminology_consistency, fluency, style_register, formatting

**Summarization criteria**: salience, coverage, faithfulness, attribution_specificity, entity_number_fidelity, coherence, conciseness, non_contradiction

**Extraction criteria**: schema_format, no_hallucinations, span_grounding, type_correctness, completeness_recall, dedup_normalization, boundary_precision, consistency

### LLM Judge Scoring

- **Judge model**: Claude 3.5 Sonnet (via OpenRouter)
- **Scoring method**: Each row is evaluated against all 8 criteria using `mlflow.genai.evaluate()` with guideline-based scorers (from `evaltune.evaluation.scorers.make_guidelines_scorer`)
- **Output**: Each criterion produces a yes/no verdict, mapped to 1/0
- **Scale**: 800 rows x 8 scorers = 6,400 LLM judge calls per task

### Error Handling

After scoring, traces are fetched from Databricks/MLflow. Rows where any scorer assessment errored are identified and dropped (e.g., 32 rows dropped from summarization, leaving 768 clean rows).

### Outputs

Saved to `./evals_benchmark_datasets/`:
- `translation_mqm_input_output_label.csv` — raw dataset (input, output, label)
- `summarization_frank_input_output_label.csv` — raw dataset
- `extraction_conll_ner_input_output_label.csv` — raw dataset
- `summarization_frank_features.csv` — dataset with 8 LLM judge feature columns appended

### Infrastructure

- **Experiment tracking**: MLflow on Databricks (experiment ID: `2241596453982194`)
- **Tracing**: `mlflow.dspy.autolog()` logs all traces, enabling inspection in the Databricks Traces UI

---

## Notebook 2: `unified_eval_learning.ipynb`

### Purpose

Trains and compares supervised ML models to predict whether a summarization output passes or fails quality review, using the 8 binary LLM judge features as input.

### Dataset

- **Source**: `summarization_frank_features.csv` (768 samples after error row removal)
- **Features**: 8 binary columns (the LLM judge criterion scores)
- **Target**: `label` (0 = FAIL, 1 = PASS)
- **Class balance**: 34.8% PASS / 65.2% FAIL

### Models Compared

| Model | Key Config |
|---|---|
| Logistic Regression (no regularization) | `class_weight='balanced'` |
| Logistic Regression (L1/Lasso) | `C=1.0, solver='saga'` |
| Logistic Regression (L2/Ridge) | `C=1.0, solver='lbfgs'` |
| Decision Tree | `max_depth=4, class_weight='balanced'` |
| Random Forest | `n_estimators=200, max_depth=5, class_weight='balanced'` |
| Gradient Boosted Trees | `n_estimators=200, max_depth=3, lr=0.1` |
| Neural Network (MLP) | `hidden_layers=(32, 16), early_stopping=True` |

**Baselines**:
- Average of all 8 features (threshold at 0.5)
- Salience feature alone

### Evaluation Setup

- **Split**: Stratified 80/20 train/test (614 train, 154 test)
- **Cross-validation**: Stratified 5-fold on training set
- **Metrics**: Accuracy, Precision, Recall, F1, ROC AUC
- **Statistical tests**: Paired t-tests across CV folds

### Results

**Cross-validation F1 scores**:

| Model | F1 (mean +/- std) |
|---|---|
| Random Forest | 0.829 +/- 0.035 |
| Decision Tree | 0.828 +/- 0.030 |
| Logistic (all variants) | 0.827 +/- 0.031 |
| Neural Net (MLP) | 0.825 +/- 0.043 |
| Gradient Boosted | 0.804 +/- 0.052 |
| Baseline (Avg Features) | 0.801 +/- 0.026 |
| Baseline (Salience) | 0.508 +/- 0.085 |

No statistically significant differences between the top ML models (paired t-test, p > 0.05 for all pairwise comparisons). The salience-only baseline is significantly worse (p < 0.001).

**Best model test-set performance (Random Forest)**:

| Metric | Score |
|---|---|
| F1 | 0.833 |
| ROC AUC | 0.922 |
| Accuracy | 0.870 |
| Precision (PASS) | 0.758 |
| Recall (PASS) | 0.926 |

### Explainability Analysis

The notebook includes:
- **Logistic regression coefficients and odds ratios** — non_contradiction has the highest odds ratio (9.81x)
- **Decision tree visualization** — full tree plot at max_depth=4
- **Feature importances** (Gini) for all tree-based models
- **Partial dependence plots** for Random Forest and Gradient Boosted
- **Permutation importance** for the Neural Network
- **Unified importance heatmap** — normalized importances across all model types
- **Error analysis** — feature profiles for false positives vs true negatives, false negatives vs true positives

### Most Predictive Features (Median Normalized Importance Across All Models)

1. **non_contradiction** (1.000) — whether the summary contradicts the source or itself
2. **faithfulness** (0.490) — whether all claims are supported by the source
3. **entity_number_fidelity** (0.453) — whether names, numbers, dates match the source

---

## Key Findings

1. **Non-contradiction dominates**: Across all model types, whether the summary contradicts the source is the single most predictive feature of overall quality, with the highest correlation (+0.726) and odds ratio (9.81x).

2. **Simple models match complex ones**: With only 8 binary features, logistic regression performs comparably to random forests and neural networks. The feature space is well-structured enough that model complexity provides minimal benefit.

3. **LLM judge features are effective**: A lightweight classifier on 8 binary LLM judge features achieves F1 = 0.833 and AUC = 0.922 on held-out data, demonstrating that per-criterion LLM assessments can reliably predict overall quality without running a full end-to-end quality judge.

4. **Baselines provide context**: Averaging all features (F1 = 0.801) is a reasonable heuristic, but any single feature alone (salience F1 = 0.508) is insufficient, confirming that the multi-criterion approach adds value.
