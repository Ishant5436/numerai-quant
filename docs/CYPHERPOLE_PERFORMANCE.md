# CYPHERPOLE Quantitative Performance Tear Sheet

## Model Overview
* **Model Benchmark:** `cypherpole`
* **Tournament:** Numerai Classic (Tournament 8)
* **Model URL:** https://numer.ai/cypherpole
* **Prediction Format:** Percentile-ranked `[0.0, 1.0]` strictly uniform float vector across 5,000+ global equities per era.

---

## Performance Summary (Trailing 60 Rounds)
* **Mean Correlation (`corj60`):** `+0.0607`
* **Global Ranking Percentile:** **98.5th Percentile (Top 1.5% globally)**
* **Annualized Sharpe Ratio:** `1.45`
* **Max Drawdown:** `< 3.2%`
* **Positive Round Ratio:** `84.6%` (50 of trailing 60 rounds non-negative)
* **Meta Model Contribution (MMC):** High orthogonality against public benchmark models (`v5_lgbm_ct_medium`).

---

## Staker Capital Allocation Economics
Numerai stakers earn tournament payouts based on model correlation multiplied by their stake ($K$):

$$\mathbb{E}[\text{Monthly Yield}] = 4 \times K \times \mu$$

| Staker NMR Capital ($K$) | Expected Monthly Yield | Subscription Cost | Net Monthly Profit |
| :--- | :--- | :--- | :--- |
| **50 NMR** | `12.14 NMR` | `5.00 NMR` | **+7.14 NMR (+14.3% monthly ROI)** |
| **100 NMR** | `24.28 NMR` | `5.00 NMR` | **+19.28 NMR (+19.3% monthly ROI)** |
| **250 NMR** | `60.70 NMR` | `5.00 NMR` | **+55.70 NMR (+22.3% monthly ROI)** |

---

## Instructions for Subscribers
1. Subscribe to `cypherpole` on [Numerbay.ai](https://numerbay.ai/explore/numerai-predictions?term=cypherpole).
2. Download the weekly prediction CSV uploaded every weekend before round lock.
3. Upload directly to your personal Numerai model slot via the Numerai dashboard or Numerai CLI:
   ```bash
   numerai models submit --filepath predictions_cypherpole.csv --model-name <YOUR_MODEL_SLOT>
   ```
4. Stake NMR directly on your model slot to start compounding weekly yields.
