Evaluation for Financial Machine Learning

A point-in-time study of whether machine-learning gains in U.S. equity panels survive a previously unseen market period.

[Open the interactive dashboard](https://evaluation-for-financial-machine-learning.streamlit.app/)

## What this project asks

Financial models can look strong after researchers repeatedly inspect the same test years. This project creates a clearer evidence boundary: develop on an exploratory period, freeze the complete experiment, and evaluate once on a later locked period.

The study is an evaluation framework and empirical comparison. It does not introduce a new prediction architecture or claim a deployable trading strategy.

The experiment enforces four rules:

1. Every feature and label is checked against when it became available.
2. Models train only on information available before each forecast.
3. The model menu, metrics, seeds, splits, and inference procedure are recorded before the locked run.
4. Final claims use paired monthly evidence, moving-block uncertainty intervals, and family-wise correction where prespecified.

## Tasks and models

Each task uses its own target, credible baseline, model menu, and primary metric. The eight-model benchmark belongs to the variance task; return ranking and delisting use smaller prespecified model sets.

| Task | Forecast | Models | Primary metric | Baseline |
|---|---|---|---|---|
| Realized variance | Next-month price variance | Persistence, HAR, Ridge, LightGBM, XGBoost, CatBoost, ExtraTrees, MLP | QLIKE, lower is better | HAR |
| Return ranking | Next-month total return | Zero, one-month reversal, Ridge, LightGBM | Mean monthly rank IC, higher is better | One-month reversal |
| Adverse delisting | Event within 12 months | Historical base rate, logistic regression, LightGBM, XGBoost | Mean monthly log loss, lower is better | Historical base rate |

The prediction universe is U.S. common equity listed on the NYSE, AMEX, or Nasdaq. The source period is 1990-2024.

## Evaluation design

| Task | Exploratory period | Locked period | Exploratory stock-months | Locked stock-months |
|---|---:|---:|---:|---:|
| Realized variance | 2000-Nov. 2021 | 2022-Nov. 2024 | 1,130,421 | 144,641 |
| Return ranking | 2000-Nov. 2021 | 2022-Nov. 2024 | 1,133,543 | 145,243 |
| Adverse delisting | 2000-2020 | 2022-2023 | 1,090,964 | 102,368 |

Models use expanding historical training data, a separate validation window, and annual walk-forward refits. The delisting task includes a 2021 embargo because its labels require a full 12-month horizon.

Primary comparisons use monthly paired differences and a circular moving-block bootstrap with 12-month blocks. The variance family also uses a Holm correction across the prespecified comparisons. Blocks of 3, 6, and 24 months are retained as sensitivity checks.

## Main locked results

Differences below are model minus baseline. Negative differences improve QLIKE or log loss; positive differences improve rank IC.

| Task | Comparison | Difference | 95% interval | Conclusion |
|---|---|---:|---:|---|
| Variance | Ridge vs. HAR | -1.8047 | [-4.1193, -0.5613] | Improvement |
| Variance | ExtraTrees vs. HAR | -1.3258 | [-2.9647, -0.4064] | Improvement |
| Variance | XGBoost vs. HAR | -0.6550 | [-1.5666, -0.0343] | Does not survive Holm correction |
| Variance | LightGBM vs. HAR | -0.2079 | [-0.8709, 0.4280] | Inconclusive |
| Returns | LightGBM vs. reversal | +0.0512 | [-0.0131, 0.1003] | Inconclusive |
| Delisting | XGBoost vs. base rate | -0.0484 | [-0.0585, -0.0384] | Improvement |
| Delisting | LightGBM vs. base rate | -0.0473 | [-0.0557, -0.0389] | Improvement |
| Delisting | Logistic vs. base rate | +0.0081 | [-0.0156, 0.0315] | Inconclusive |

The locked results are deliberately mixed. Linear shrinkage and randomized trees improve variance forecasts, boosted trees improve delisting probability forecasts, and the apparent return-ranking gain remains too uncertain for a reliable claim. Model choice depends on the prediction target and scoring rule.

## Dashboard

The hosted Streamlit app reads the compact aggregate artifacts in [`dashboard_data/`](dashboard_data/). It does not retrain models when a visitor changes a selection.

The app provides:

- exploratory and locked results for all three tasks;
- all saved models within each task, including the eight-model variance benchmark;
- pairwise model comparisons on the same saved months;
- turnover and simple transaction-cost diagnostics for return forecasts;
- saved factor-reliance results where available; and
- an external-prediction evaluator for a user's own dataset.

To evaluate external predictions, upload a CSV or Parquet file with at least two numeric columns: the realized outcome and the model forecast. Return forecasts can also include a month or date column. The app scores the uploaded predictions; it never executes uploaded model code.

## Run the dashboard locally

Python 3.11 is recommended.

```bash
git clone https://github.com/Atharva-099/Locked-Evaluation-for-Financial-Machine-Learning.git
cd Locked-Evaluation-for-Financial-Machine-Learning

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r app/requirements.txt

streamlit run app/Home.py
```

This lightweight installation is sufficient for the dashboard because its model outputs are already computed.

## Full research environment

Install the complete dependency set to work with the training, evaluation, and reporting code:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python -m p3_modellab --help
```

Full replication requires licensed CRSP data through WRDS, the project data contract, and substantial CPU and memory. Raw licensed records, full panels, row-level forecasts, and fitted model objects are not redistributed. The public `dashboard_data/` bundle contains only the aggregate artifacts needed to inspect the reported results.

Quick runs validate code paths only. They are not evidence for the empirical conclusions.

## Repository structure

```text
app/                Streamlit dashboard
dashboard_data/     Compact saved artifacts used by the hosted app
experiments/        Reproducible stage and cloud entry points
p3_modellab/        Data contracts, models, runners, metrics, and reports
tests/               Unit and integration checks
```


The study evaluates predictive performance under a locked temporal protocol. The return portfolio is a diagnostic rather than a deployable backtest: market impact, capacity, borrow availability, short-sale fees, and operational constraints remain outside the main claim.

Results depend on the stated universe, time period, feature set, targets, and model menus. The locked period reduces researcher adaptation to the final years, but it does not make the findings universal across markets or future regimes.
