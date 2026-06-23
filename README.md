### Predicting Helpfulness and Harmlessness of LLM Responses

#### Executive summary

Evaluating whether an LLM's response is helpful and safe currently depends on slow, expensive human
annotation. This project asks whether a model can learn to predict both qualities directly from features of
the response itself. Using Anthropic's HH-RLHF human-preference dataset, this first phase builds an
interpretable baseline: a logistic regression that predicts, from surface-level text features alone (length,
vocabulary richness, hedging language, refusal phrasing), whether a response was the one humans preferred.
The baseline beats chance on both helpfulness and harmlessness, but only modestly — confirming that
surface-level signals carry real information while leaving most of the predictive power to the
embedding-based and transformer models planned next. It also surfaces a genuinely useful
finding: the features that predict a "good" response point in *opposite directions* depending on whether
the goal is helpfulness or harmlessness, which is exactly the tension a joint helpfulness/harmlessness model
needs to resolve.

#### Rationale

Every major AI lab invests heavily in automated reward modeling — systems that score LLM outputs to guide
RLHF training and to catch quality or safety regressions after deployment, without needing a human in the
loop for every judgment. A lightweight, explainable model that scores both helpfulness and harmlessness and
attributes its scores to specific response features would be directly useful in production AI evaluation
pipelines, and would make those pipelines auditable rather than a black box.

#### Research Question

Given a conversational prompt and a candidate LLM response, can a machine learning system predict both (1) a
helpfulness score and (2) a harmlessness score for the response, and identify which response features drove
each score — without requiring a human label at inference time?

#### Data Sources

[Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) (Helpful and Harmless RLHF), released
under an MIT license on Hugging Face and loaded directly via the `datasets` library. It contains roughly
161,000 human preference pairs: a conversational prompt plus two candidate responses, with a human
annotator's choice of the preferred one. This phase uses the `helpful-base` and `harmless-base` subsets
(~92K and ~90K response rows respectively, using Anthropic's original train/test split). The `helpful-online`
and `helpful-rejection-sampled` subsets are planned for a future phase.

#### Methodology

This phase covers an interpretable baseline layer plus the EDA that precedes it:

1. **Data cleaning** — checked for missing values, malformed parses (a small number of records, well under
   0.1%, failed to parse a final Assistant turn), and duplicate records (2 exact duplicates found and
   dropped). Several thousand rows share identical short response text (e.g. repeated refusal phrases); these
   were kept, since they reflect real recurring model behavior rather than corrupted data.
2. **Feature engineering** — derived interpretable, surface-level features per response: character/word/
   sentence counts, average word length, vocabulary richness (unique/total word ratio), a hedge-word ratio,
   a refusal-phrase detector, question/exclamation counts, a response-to-prompt length ratio, and an
   approximate Flesch Reading Ease score.
3. **Exploratory data analysis** — class balance, response-length distributions (chosen vs. rejected, by
   subset), IQR-based outlier analysis, hedging/refusal rate comparisons, and a feature correlation heatmap.
4. **Baseline model** — a logistic regression (on standardized features) trained separately for
   `helpful-base` and `harmless-base`, predicting whether a given response was the one annotators chose.
   Evaluated with accuracy, ROC-AUC, and F1 on Anthropic's held-out test split.

Planned next: semantic embedding features and a fine-tuned dual-head transformer reward model, then an
LLM-as-judge comparison layer — see Next Steps.

#### Results

**In plain terms:** text alone — no AI judgment, just things like length and whether the reply refuses the
request — can correctly guess which of two responses a human preferred about 56-57% of the time, well above
a coin flip. That's not enough to replace human review on its own, but it's a useful, fast, fully
explainable first filter, and it already reveals that "helpful" and "harmless" reward very different kinds
of writing.

- The baseline reaches **56.7% accuracy / 0.598 ROC-AUC** on `helpful-base` and **55.9% accuracy / 0.570
  ROC-AUC** on `harmless-base` — both clearly above the 50% chance baseline (classes are exactly balanced by
  construction), but modest in absolute terms. Surface text features alone carry a real but limited
  preference signal.
- **Helpfulness** preference leans toward longer, more varied responses: length features
  (character/word count, response-to-prompt ratio) have positive coefficients, and low vocabulary richness
  (`unique_word_ratio`) is the single strongest *negative* predictor — repetitive answers are disfavored.
- **Harmlessness** preference behaves differently, and in places oppositely: refusal phrasing positively
  predicts being chosen (annotators reward safe refusals), while response length is the strongest negative
  predictor — long-winded responses to sensitive prompts are disfavored, the reverse of the helpfulness
  pattern. A single feature set behaving oppositely across the two objectives is itself a key finding, and
  motivates the multi-task model formulation planned next.
- Data quality is high: under 0.1% of records were dropped during cleaning, and roughly 5-6% of responses
  are length outliers (mostly long, detailed answers) — these were kept, since standardized linear modeling
  handles the resulting skew without discarding real responses.
- Full analysis, code, and visualizations are in the notebook linked below.

#### Next steps

- Add sentence-BERT embedding features and prompt-response cosine similarity to capture meaning beyond
  surface statistics, and compare against this baseline.
- Fine-tune a transformer-based dual-head reward model with a combined helpfulness/harmlessness loss, and
  run ablations on the loss-weighting tradeoff.
- Build an XGBoost/LightGBM multi-task ensemble with SHAP feature attribution for per-prediction
  explanations, not just global coefficients.
- Add an LLM-as-judge comparison layer as an independent zero-shot baseline.
- Extend to the `helpful-online` and `helpful-rejection-sampled` subsets for a fuller picture of helpfulness.

#### Outline of project

- [Data Preparation, EDA, and Baseline Model](/01_data_preparation_eda.ipynb)

#### Requirements

The notebook is self-contained and downloads its data automatically via the Hugging Face `datasets`
library — no manual data download needed. To run it locally:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn datasets jupyter
jupyter notebook notebooks/01_data_preparation_eda.ipynb
```

##### Contact and Further Information

For questions about this project, please open an issue on this repository.
