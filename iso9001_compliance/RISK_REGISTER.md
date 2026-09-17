# Software Quality Risk Register (FMEA Matrix): Numerai Quant
## Conforming to ISO/DIS 9001:2026 Clause 6 (Risk-Based Thinking)

This document tracks identified quantitative risks, statistical failure modes, and automated mitigations for `numerai-quant`.

---

## 1. Risk Evaluation Scale
- **Severity (S):** 1 (Negligible) to 5 (Tournament score wipeout / burned stake)
- **Likelihood (L):** 1 (Extremely Rare) to 5 (Frequent without controls)
- **Risk Priority Number (RPN):** $S \times L$ (Scale 1 to 25). RPN $\ge 12$ mandates automated gating.

---

## 2. Failure Modes and Effects Analysis (FMEA)

| Risk ID | Potential Failure Mode | Impact / Effect | Severity (S) | Likelihood (L) | Initial RPN | Automated Mitigation & Quality Control | Residual RPN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSK-NQ-01** | NaN values or corrupted values in tournament submission file | Immediate round disqualification / submission rejection | 5 | 4 | **20** | Strict `validate_prediction_file` asserting zero NaNs and $[0.0, 1.0]$ uniform bounds; verified in `test_numerbay_publisher.py` & `test_whitebox.py` | **2** (S=2, L=1) |
| **RSK-NQ-02** | Collinear signal submission cannibalizing Meta Model Contribution | Negative MMC payout penalty from hedge fund | 5 | 3 | **15** | Tri-hurdle orthogonality test asserting pairwise correlation bounds ($<0.70$); verified in `test_orthogonality.py` & `test_fleet_30.py` | **2** (S=2, L=1) |
| **RSK-NQ-03** | Broad market regime drawdown affecting equity factor models | Severe portfolio Sharpe collapse | 4 | 4 | **16** | Closed-form QR decomposition feature neutralization projecting out principal risk factors; verified in `test_whitebox.py` | **2** (S=2, L=1) |
| **RSK-NQ-04** | Missing serialized LightGBM weights during weekly submission | Submission pipeline crash / missed deadline | 5 | 3 | **15** | Pre-flight model file existence assertions; verified in `test_fleet_60d_model_files_exist` and `test_blackbox_production_fails_loudly_when_models_missing` | **2** (S=2, L=1) |
| **RSK-NQ-05** | API connection timeout during weekend submission window | Missed round lock-in | 4 | 3 | **12** | Exponential backoff retry loop with verified S3 upload status confirmation; verified in `test_blackbox_safe_upload_predictions_verifies_s3_status` | **2** (S=2, L=1) |
| **RSK-NQ-06** | Memory leak or stack overflow in Chimera genetic formula evaluation | Process crash during genetic synthesis | 4 | 3 | **12** | Native C++ Chimera arena memory allocator with maximum AST depth ceiling ($\le 8$); verified in `test_chimera_eval.py` & `test_ast_depth_constraint` | **2** (S=2, L=1) |

---

## 3. Governance
Audited on each commit via `make audit-iso9001` and continuous integration.
