# Evaluation protocol

Run `python scripts/baseline_eval.py --seed 42` from the repository root after installing `requirements.txt`. The output is `artifacts/baseline_eval.json`. Run `python -m pytest tests -q` for regression checks.

## Boundaries

- The evaluator locks A2 and A5 attack types out of training, fits on earlier simulation steps, and scores later steps/fresh attack executions.
- The meta-model uses out-of-fold base predictions. Its XGBoost-only baseline is evaluated on identical rows. Ensemble superiority is not assumed.
- Precision, recall, F1, confusion counts, and cost use a predeclared threshold of 0.5. If tuning a threshold, select it on a separate validation set before final testing. The reporting harness no longer optimizes against test labels.
- PR-AUC means trapezoidal PR-curve integration, not average precision. A single-class population has no reported discriminative PR-AUC.
- Per-attack recall above evaluation normals' p95 remains an exploratory diagnostic, not an operational threshold estimate.

## Prevalence

The declared policy retains all fraud and subsamples normal rows without replacement (seed 42). Requests that need more normal rows than are available are marked unavailable, without a PR-AUC. Available outputs record requested and achieved prevalence and actual counts.

The old 5% headline used a roughly 43% fraud population. It must not be reused as low-prevalence evidence. Duplicating normal rows would not create independent evidence. Any alternative sampling policy must be documented separately.

## Reproducibility and limits

The JSON records seed, configuration, runtime, holdout fingerprint, train/evaluation counts, cut step, and both models' results. Results depend on the generator; held-out attack types and fresh seeds do not establish real-bank generalization. Review feature causality, graph construction and account overlap independently. Use multiple seeds and confidence intervals for research claims. Cost assumptions FP=1/FN=10 are declared inputs, not measured financial returns.

[CI runs](https://github.com/KA-1205/prometheus/actions) provide test reports. A keep-alive ping is not a test run.

## Recorded verification (17 October 2026)

The fresh seed-42 default run evaluated 7,424 later-step rows (129 fraud; about 1.74%). Its 5% resampled population retains 129 fraud rows and contains 2,580 total rows, so requested and achieved prevalence agree. Overall held-out meta PR-AUC was about 0.8628 versus XGBoost-only 0.8106. These are one-run synthetic results; the per-attack diagnostic still misses A2, so broad generalization claims are not warranted. Inspect the full JSON rather than quoting the aggregate alone.
