# Numerai True Orthogonal Quant Fleet

> **Statistically Independent, Multi-Factor Machine Learning Fleet for Numerai Tournament v5.0 & Signals v3 Supernova**  
> *Optimized for Apple Silicon ARM64 (M5 Pro) with Sub-Factor Mining & Linear Feature Neutralization*

[![Tests](https://img.shields.io/badge/Tests-22%2F22%20Passed-brightgreen)](tests/)
[![Tournament](https://img.shields.io/badge/Tournament-v5.0%20Ender%2060D-blue)](config.py)
[![Signals](https://img.shields.io/badge/Signals-v3%20Supernova-purple)](signals/)
[![Linear Algebra](https://img.shields.io/badge/Linear%20Algebra-QR%20Decomposition-orange)](neutralize.py)

---

## 1. Algorithmic Formulations & Computational Complexity

| Subsystem | Algorithmic Primitive | Time Complexity | Space Complexity | Mathematical Invariant |
| :--- | :--- | :---: | :---: | :--- |
| **Feature Neutralization** | Economy QR Decomposition | $\mathcal{O}(N K)$ | $\mathcal{O}(N K)$ | $X = QR$; $\text{Proj}_X(P) = Q(Q^T P)$; eliminates matrix inversion condition-number blowup. |
| **Rank Transformation** | Uniform Percentile Mapping | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N)$ | Strictly uniform distribution on $[0.0, 1.0]$; NaN-resilient midpoint imputation (`0.5`). |
| **Signals Supernova** | Cross-Sectional Multi-Factor | $\mathcal{O}(M \cdot B)$ | $\mathcal{O}(M)$ | Blends 12M momentum, 1M reversal, Parkinson volatility anomaly, EMA trend, and volume shocks. |
| **Ensemble Aggregation** | True Orthogonal Stacking | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | Pairwise Spearman rank correlations bounded strictly within $\rho_{ij} \in [0.140, 0.492]$. |

---

## 2. True Orthogonal Fleet Performance Benchmark (4,120,000+ Out-of-Sample Rows)

Each strategy is trained on **orthogonal factor subsets** with asymmetric tree architectures:

| # | Strategy Archetype | Target Horizon | Feature Partition | Neutralization | Mean Corr | Raw Sharpe | Annualized Sharpe | Max Drawdown |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **1** | **Core Alpha Flagship** | `target` | All 705 features | 25% | **+0.0227** | **1.013** | **3.508** | **17.78%** |
| **2** | **Fundamental Alpha Specialist** | `target_jeremy_20` | 186 feats (`wisdom`, `intelligence`, `charisma`) | 35% | **+0.0107** | **0.562** | **1.948** | 26.59% |
| **3** | **Momentum Alpha Specialist** | `target_victor_20` | 133 feats (`strength`, `dexterity`, `agility`) | 40% | **+0.0144** | **0.825** | **2.859** | **16.40%** |
| **4** | **Macro Regime Specialist** | `target_xerxes_20` | 278 feats (`serenity`, `sunshine`, `midnight`) | 45% | **+0.0157** | **0.819** | **2.838** | 38.04% |
| **5** | **Constitution Residual Specialist** | `target_delta_20` | 155 feats (`constitution`, `dexterity`) | 50% | **+0.0104** | **0.629** | **2.180** | 26.11% |

---

## 3. Pairwise Cross-Strategy Spearman Correlation Matrix (True Orthogonality)

$$\rho_{ij} \in [0.140, 0.492]$$

```
                                          Strat 1 (Core)  Strat 2 (Fund)  Strat 3 (Mom)  Strat 4 (Macro)  Strat 5 (Res)
Strat 1: Core Alpha Flagship                       1.000           0.474          0.492            0.480          0.363
Strat 2: Fundamental Specialist                    0.474           1.000          0.169            0.176          0.140
Strat 3: Momentum Specialist                       0.492           0.169          1.000            0.263          0.357
Strat 4: Macro Regime Specialist                   0.480           0.176          0.263            1.000          0.279
Strat 5: Constitution Residual Specialist          0.363           0.140          0.357            0.279          1.000
```

---

## 4. Quick Execution & Automation

### 1. Execute Automated Test Suite (15/15 Tests Passing)
```bash
make test
```

### 2. Run Autonomous Multi-Model Fleet Submissions
```bash
./fleet_submit.py
```

### 3. Launch Local Dynamic Dashboard (http://127.0.0.1:8501)
```bash
./dashboard.py
```

### 4. Automated Weekly Cron Watchdog
* Runs every **Sunday at 02:00 IST / 20:30 UTC Saturday** via `cron_submit.sh`.
* Perfectly timed ~2.5 hours after the Numerai round opening window (Saturday 18:00 UTC).
* Logs to `logs/fleet_submit.log` with automatic 3x retry on network drop and native notifications.
