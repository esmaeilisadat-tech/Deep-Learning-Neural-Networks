# Day 9 RNN/LSTM — Extend the Existing Forecasting Exercise into a Job-Ready Case Study

**Source audit:** 24 Nov 2026. **Status: Scope and acceptance plan ONLY**; no TensorFlow training, dataset import, new model results or cloud deployment was performed.

## What is already implemented and must be preserved

The existing [Intermediate notebook](notebooks/Recurrent_Networks_LSTM_Intermediate.ipynb) has three teaching exercises. Exercise 2 builds a synthetic 1,000-observation stock-like sequence (trend + sine seasonality + random walk), forms **30-step input / 5-step targets**, trains a three-layer LSTM, and calculates test-set MAE in original units. Exercise 3 compares approaches on a noisy sine series. Original Beginner/Intermediate/Advanced notebooks are educational evidence; **do not rewrite or overwrite them**.

These are **synthetic teaching exercises**, not industrial or stock-trading forecasting evidence. The older [ML Engineering Portfolio](https://github.com/esmaeilisadat-tech/ML_Engineering_Portfolio-) also includes steel-energy ensemble exercises, but they are distinct from this five-step sequence model.

## Exact known issue to fix in a NEW adjacent extension

In Exercise 2 the sequence is normalized as follows **before** train/test separation:

```python
kurs_min, kurs_max = kursdaten.min(), kursdaten.max()
kursdaten_norm = (kursdaten - kurs_min) / (kurs_max - kurs_min)
```

This uses future test-set extrema for the training transform, a form of **look-ahead leakage**. The code also creates overlapping window targets before splitting. The extension must:

1. Choose immutable train/validation/test cutoff **timestamps** before fitting any scaler. Fit transformations only on the training fold, apply to subsequent folds.
2. For every rolling origin, assert `max(training_target_timestamp) < first_validation_target_timestamp`, and exclude any window whose target crosses the train/validation boundary. Evaluate only forecasts issued with past inputs.
3. Preserve an untouched final test period; choose model configuration and interval calibration on train/validation only.

## Applied extension — use ONE approved demand dataset

Choose a licensed/approved retail weekly-unit-demand dataset or an industrial energy-consumption series with a genuine **continuous numerical target** and timestamped observations. Document source/license, version, date range, frequency, missing timestamps, known future covariates and release cutoff. Do not mislabel the existing sine/stock-like synthetic teaching data as actual retail sales.

**Avoid unnecessary model proliferation.** Prefer three core comparators: seasonal naive, one transparent statistical baseline (ETS or SARIMA), and one supervised tabular model (LightGBM **or** XGBoost). The existing LSTM may be included as a fourth comparator only when time, capacity and data size support a fair comparison. The purpose is **forecasting benchmark quality**, not maximizing framework count.

## Reproducible validation

- **Scope:** weekly prediction, four-week horizon, and at least six rolling-origin evaluations when there is sufficient history; reselect horizon only through a documented decision.
- **Splits:** chronological by issuance timestamp, no future target leakage, fold-local imputation/scaling and no testing on nonmatured labels.
- **Metrics:** MAE, RMSE and WAPE/MASE (record zero-demand denominator handling); compare each horizon and origin against seasonal naive.
- **Uncertainty:** p10/p50/p90 quantiles, pinball loss, nominal 80% interval coverage and mean interval width; describe how quantiles are trained/calibrated without future labels.
- **Error analysis:** low-volume products, outliers, seasonality, holidays, missing history, regime changes and systematic bias.
- **Batch service:** CLI `--as-of` and `--horizon` on a fixed fixture, no internet/provider calls, deterministic output schema and model artifact hash.
- **Quality:** unit tests for cutoff/overlap/leakage, missing weeks, scaling train-only, nonnegative output policy if applicable, identical reruns, fail-closed input validation and serializable output.
- **CI:** minimal offline unit/integration checks independent of TensorFlow training; no paid cloud workflow by default.

## Acceptance / definition of done

- [ ] The original 45-course-notebook repository remains intact; no mass renaming/deletion.
- [ ] Dataset license and timestamp integrity are recorded and reproducible from an approved source.
- [ ] Target, forecasting horizon, rolling origins and baseline metrics are fixed before model comparison.
- [ ] One full offline backtest and an independent holdout evaluated without look-ahead leakage.
- [ ] Baselines and model results reported with interval-coverage diagnostics and failure cases; no invented improvement.
- [ ] A README links to a small runnable `forecasting_extension/` module/notebook and a compact verified batch sample.
- [ ] Reproducible installation, tests and optional CI (activated separately only after owner approval).
- [ ] English/German project summary distinguishes synthetic course notebook from the new applied extension.

**Do not claim the proposed applied demand forecasting system already exists, is trained, was run or is deployed.** This document identifies the precise reusable course project and how to complete it rather than creating another unrelated repository.
