# Project Prometheus

**Synthetic fraud-detection research prototype**

A stateful Financial Digital Twin generates synthetic transactions with 8 AMLSim-verified fraud typologies. A rule-based Attack Compiler produces 6 benchmark attack types (2 held out for generalization testing). The Blue Team (XGBoost + GNN + meta-model) detects fraud, and a Sensitivity Engine diagnoses misses to drive weakness-directed retraining (max 2 rounds). A FastAPI dashboard exposes attack, diagnosis, and bounded retraining flows. Whether retraining helps is evaluated separately; success is not assumed.

This is Kartik Arora’s fork of [harshil-sri/prometheus](https://github.com/harshil-sri/prometheus), a collaborative project. My documented contributions cover the SSE fan-out/dashboard stream and tabular-diffusion fidelity evaluation. See [authored commits and tests](docs/CONTRIBUTIONS.md).

[Evaluation protocol](docs/EVALUATION.md) · [Raw evaluation](artifacts/baseline_eval.json) · [Tests](tests/) · [CI runs](https://github.com/KA-1205/prometheus/actions)

## Recorded result and limits

The recorded seed-42 run evaluated 7,424 later-step synthetic rows, including 129 fraud rows (about 1.74%). Meta-model PR-AUC was **0.8628**, compared with **0.8106** for XGBoost-only on the same held-out rows. This is one generator-specific run, not real-bank validation. The per-attack diagnostic still misses A2; broad generalization claims are unsupported. See the [protocol](docs/EVALUATION.md) for fixed thresholds, prevalence, and data boundaries.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run temporal held-out evaluation; never report training-set fit as test performance
python scripts/baseline_eval.py --seed 42
# Compare meta-model and XGBoost-only on the SAME held-out rows.
python -m pytest tests -q

# Start the dashboard
cd src/api && PYTHONPATH=.. python3 main.py
# Open http://localhost:8000
```

## Project Structure

```
src/
├── twin/           # Financial Digital Twin (T1)
│   ├── core.py     # Entity schemas + WorldState
│   ├── normal_behavior.py  # Normal transaction generator
│   ├── typologies.py       # 8 AMLSim typologies
│   └── twin.py     # FinancialDigitalTwin orchestrator
├── attack/         # Attack Compiler (T2)
│   ├── compiler.py # Rule-based planner
│   ├── spec.py     # Attack spec schema
│   └── benchmark_attacks.py  # 6 attack types (A2/A5 held out)
├── blue/           # Blue Team (T3)
│   ├── features.py # Feature engineering
│   ├── xgb_model.py    # XGBoost detector
│   ├── gnn_model.py    # 2-layer SAGEConv GNN
│   ├── meta_model.py   # Logistic blend
│   └── calibrate.py    # Platt/isotonic calibration
├── sensitivity/    # Sensitivity Engine (T4)
│   ├── engine.py       # Shared engine (3 consumers)
│   ├── shap_explainer.py  # SHAP for XGBoost
│   └── gnn_ablation.py    # Masked-neighbor ablation
├── feedback/       # Feedback Loop (T5)
│   ├── loop.py     # Weakness-directed retrain (≤2 rounds)
│   └── report.py   # Blind-Spot Report
├── scoring/        # Structured Score (T6)
│   └── structured_score.py  # 0-1000 Mastercard bands
├── eval/           # Evaluation Harness (T8)
│   └── harness.py  # Multi-prevalence PR-AUC, cost model
├── api/            # Dashboard Backend (T7)
│   ├── main.py     # FastAPI server
│   └── graph.py    # Multi-relational Knowledge Graph engine
└── dashboard/      # Dashboard Frontend (T7)
    └── index.html  # Single-page War-Room UI + Graph Canvas
```

## Key Design Decisions

- **Knowledge Graph Explorer**: Dynamic extraction of multi-relational graphs (accounts, customers, merchants, devices, IPs, transactions) with GNN risk overlay and sub-graph trajectory filtering without hardcoded constants
- **Held-out attacks**: A2 (synthetic identity) and A5 (scatter_gather layering) are locked before training — enforced in code with `assert_no_held_out_leakage()`
- **Max 2 retrain rounds**: Hard-capped in `feedback/loop.py` to prevent overfitting to generator quirks
- **Structured score is deep-path only**: Fast path stays a pure ML probability. Never mixed.
- **Deterministic**: Every component seedable via `random.Random(seed)`
- **Cold-start**: Accounts with no history produce sane default features (no NaNs)
- **Isolated nodes**: GNN handles nodes with no edges via self-loop normalization

## Demo

Use the dashboard to inspect three stages:

1. Generate attacks and inspect which ones the current detector misses.
2. Diagnose those misses and run at most two weakness-directed retraining rounds.
3. Evaluate held-out attack types A2 and A5, retaining failures as well as improvements.

The demo illustrates the workflow. Use the held-out evaluator and raw artifacts for performance claims.

## Graded Artifacts

1. **Runnable GitHub repo** — this repository
2. **.docx walkthrough** — [Prometheus_Walkthrough.docx](docs/archive/Prometheus_Walkthrough.docx) (covers novel attacks, generation/simulation, detection model + efficacy, real-world feasibility)
3. **Working web prototype** — `http://localhost:8000` (FastAPI + single-page HTML/JS)

## Evaluation protocol

See [EVALUATION.md](docs/EVALUATION.md) for data boundaries, fixed thresholds, prevalence, baselines, and limitations. Synthetic results do not establish real-bank performance.

## My contributions

See [the contribution record](docs/CONTRIBUTIONS.md) for account-linked implementation and test changes.
