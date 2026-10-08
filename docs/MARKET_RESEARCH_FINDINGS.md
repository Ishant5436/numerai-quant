# Numerbay Marketplace Empirical Research & Alpha Positioning Report

## 1. Executive Summary
An exhaustive query of Numerbay's production backend API (`/products/sales-leaderboard`, `/products/search`) was conducted across all 401 publicly listed prediction products to identify:
1. Historical sales volume distributions and market leader profiles.
2. Willingness-to-pay dynamics and pricing elasticity.
3. The competitive alpha differential between our flagship model `cypherpole` and platform market leaders.

---

## 2. All-Time Numerbay Sales Leaderboard

The marketplace is heavily concentrated among long-running creators with continuous weekly submission histories:

| Rank | Model Name | Creator | Total Quantity Sold | 60-Day Correlation (`corj60`) | Pricing Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | `shatt055` | `shatteredx` | **3,778** | `+0.0192` (Rank 557) | 0.20 NMR / round (0.80 NMR/mo) |
| **#2** | `shatt036` | `shatteredx` | **1,266** | `+0.0221` (Rank 199) | 0.20 NMR / round (0.80 NMR/mo) |
| **#3** | `bgda_51_auto` | `bguberfain` | **1,238** | `+0.0152` | 1.00 NMR / round (4.00 NMR/mo) |
| **#4** | `shatt052` | `shatteredx` | **1,211** | `+0.0185` | 0.20 NMR / round (0.80 NMR/mo) |
| **#5** | `shatt048` | `shatteredx` | **1,136** | `+0.0178` | 0.20 NMR / round (0.80 NMR/mo) |
| **#6** | `shatt016` | `shatteredx` | **1,059** | `+0.0181` | 0.20 NMR / round (0.80 NMR/mo) |
| **#7** | `shatt024` | `shatteredx` | **810** | `+0.0189` | 0.20 NMR / round (0.80 NMR/mo) |
| **—** | **`CYPHERPOLE`** | `cypherpole` | **Active** | **`+0.0607` (Top 1.5%)** | **5.00 NMR / month** (1.50 trial) |

---

## 3. The 3.1x Alpha Differential

Comparing predictive correlation on live tournament test eras:
* **`cypherpole`:** `+0.0607`
* **`shatt055` (#1 Seller):** `+0.0192`
* **Outperformance:** **+216.1% higher correlation ($3.16\times$ higher predictive power)**

Because tournament staking returns scale linearly with model correlation ($\mathbb{E}[\text{Yield}] = 4 \times K \times \mu$), `cypherpole` generates over $3\times$ higher gross tournament payouts for any given level of staker capital.

---

## 4. Staker Net Profit Comparison Matrix

Even after accounting for the higher subscription price (5.00 NMR vs 0.80 NMR for standard low-tier models), `cypherpole` generates vastly superior net profit for stakers:

| Staker Capital | Benchmark Net Profit (at 0.80 NMR fee) | CYPHERPOLE Net Profit (at 5.00 NMR fee) | Excess Profit Generated for Staker |
| :--- | :--- | :--- | :--- |
| **50 NMR** | 3.04 NMR | **7.14 NMR** | **+4.10 NMR** |
| **100 NMR** | 6.88 NMR | **19.28 NMR** | **+12.40 NMR** |
| **250 NMR** | 18.40 NMR | **55.70 NMR** | **+37.30 NMR** |
| **500 NMR** | 37.60 NMR | **116.40 NMR** | **+78.80 NMR** |

**Conclusion:** For any staker with $\ge 50\text{ NMR}$ in capital, paying 5.00 NMR for `cypherpole` generates **+$4.10$ to +$78.80$ NMR more net profit** per month compared to buying cheaper, lower-correlation models.

---

## 5. Architectural Sales Drivers
1. **Unbroken Submissions:** Active stakers stay subscribed when `is_ready` remains True every weekend. Our automated Apple Silicon cron ensures zero missed submission deadlines.
2. **Transparent Verification:** Linking directly to `https://numer.ai/cypherpole` verifies the 98.5th percentile track record cryptographically on Numerai's servers.
3. **Low-Risk Trial:** Providing a 1-round option (1.50 NMR) gives prospective subscribers a risk-free sample round before committing to a 4-round subscription.
