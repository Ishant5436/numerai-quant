# Tournament Staker Integration & Submission Guide

This guide details how tournament participants can acquire, blend, and submit `cypherpole` predictions for Round 1373 and upcoming weekend rounds.

---

### 1. Alpha Profile & Empirical Track Record
* **Tournament Model:** [`numer.ai/cypherpole`](https://numer.ai/cypherpole)
* **60-Day Correlation (corj60):** **+0.0607** (Top 1.5% global rank, 84.2% win rate)
* **Architecture:** Multi-target LightGBM quintet ensemble with Probit Gaussian QR feature neutralization across canonical risk factors
* **Synergy Edge:** Strips out broad factor beta, stabilizing returns during down eras where generic market models incur burns

---

### 2. Redeeming Free Trial on Numerbay
1. Navigate to the storefront listing: [Numerbay Cypherpole Listing](https://numerbay.ai/explore/numerai-predictions?term=cypherpole) (SKU 1216).
2. Select the **Single Round Trial** option (1.5 NMR).
3. Enter coupon code: **`CYPHER100`** at checkout.
4. The price drops to **0.00 NMR** (100% off for the first 5 stakers).
5. Download your delivered `predictions_cypherpole.csv` artifact.

---

### 3. Submitting to Numerai in 60 Seconds

You can submit directly via the web portal or automate the entire workflow using our open-source ingestion script.

#### Option A: Direct Turnkey Submission (Pure cypherpole)
```bash
python scripts/buyer_quick_submit.py \
  --file predictions_cypherpole.csv \
  --model <YOUR_NUMERAI_MODEL_NAME>
```

#### Option B: Convex Blending (Your Model + 25% cypherpole)
If you already run a model and want to diversify drawdown risk:
```bash
python scripts/buyer_quick_submit.py \
  --file predictions_cypherpole.csv \
  --model <YOUR_NUMERAI_MODEL_NAME> \
  --blend path/to/your_predictions.csv \
  --weight 0.25
```
* **What this does:** Merges both prediction sets, applies rank-percentile convex averaging, and normalizes strictly to a uniform $[0.0, 1.0]$ distribution before uploading directly to your slot.

---

### 4. Technical Validation Invariants
The submission engine enforces the following invariants before touching the NumerAPI upload endpoint:
1. Exact row-count validation against active tournament targets.
2. 0 NaN values and strict value bounds within $[0.0, 1.0]$.
3. Pre-flight verification of model ownership via local NumerAPI credentials.
