# Software Quality Management System (QMS) Manual: Numerai Quant
## Conforming to ISO/DIS 9001:2026 (Draft International Standard)

---

### 1. Scope & Application
This Quality Manual formalizes the Software Quality Management System (QMS) policies, statistical controls, and automated verification protocols across `numerai-quant`. It governs quantitative hedge fund predictions, multi-target LightGBM ensembles, closed-form factor neutralization, and daily Signals v3 pipelines under **ISO/DIS 9001:2026**.

---

### 2. Clause 4: Context of the Organization & Digital Infrastructure
- **4.1 Quantitative Computing Context:** Operates on Apple Silicon ARM64 utilizing vectorized NumPy/SciPy operations, optimized LightGBM inference, and native C++ Chimera arena expression engines (`chimera/csrc/`).
- **4.2 Stakeholder Expectations:** Numerai Hedge Fund, NumerBay market buyers, and tournament participants require zero submission format errors, zero NaN predictions, uniform percentile distribution on $[0.0, 1.0]$, and robust feature neutralization.
- **4.3 Scope of the QMS:** Covers all 25 classic tournament strategies, 5 daily Signals v3 models, Chimera AST engine, genetic synthesizers, purged walk-forward cross-validation, and NumerBay marketplace integrations.
- **4.4 Automated Quality Pipeline:** Managed through [`Makefile`](file:///Users/ishantpanchal/numerai-quant/Makefile) executing 85 automated test cases across unit, integration, and statistical invariant suites.

---

### 3. Clause 5: Leadership & Quality Culture
- **5.1 Leadership & Commitment:** Enforces the **Zero Completion Claims Without Verification** policy. No model upload or tournament submission is executed without local validation and fresh hash confirmation.
- **5.2 Quality Policy:** Dedicated to out-of-sample orthogonality, MMC maximization, drawdown reduction via QR neutralization, and zero-contact automated submission.
- **5.3 Roles & Responsibilities:** Automated cron monitoring daemons, statistical invariant tests, and AST safety audits act as automated quality gatekeepers.

---

### 4. Clause 6: Planning & Risk-Based Thinking
- **6.1 Actions to Address Risks & Opportunities:** The QMS maintains an active [`RISK_REGISTER.md`](file:///Users/ishantpanchal/numerai-quant/iso9001_compliance/RISK_REGISTER.md) evaluating failure modes like feature drift, collinearity collapse, NaN prediction output, and API submission timeout.
- **6.2 Quality Objectives:**
  - *Reliability:* 100% test pass rate across 85 test targets.
  - *Distribution Invariant:* Exact uniform distribution on $[0.0, 1.0]$ with zero NaNs and proper tie handling.
  - *Neutralization:* Linear projection orthogonalization reducing feature exposure while preserving alpha.
  - *Orthogonality:* Tri-hurdle acceptance rejecting collinear signals and fluke stability.

---

### 5. Clause 7: Support & Tool Qualification
- **7.1 Resources & Qualified Compilers:**
  - Python Runtime: Python 3.12 within dedicated `.venv`/`venv`.
  - Native Compiler: Apple Clang C++ for Chimera AST arena evaluation.
  - Data Processing: NumPy, Pandas, SciPy, LightGBM.
- **7.2 Competence & Documentation:** Documented in [`README.md`](file:///Users/ishantpanchal/numerai-quant/README.md) and quantitative research specs.
- **7.5 Documented Information:** Model artifacts, S3 submission UUID receipts, and audit reports are permanently archived.

---

### 6. Clause 8: Operational Planning and Control (Software V&V)
- **8.1 Verification and Validation Protocol:**
  - *Verification (Unit & Whitebox):* Mathematical verification of `rank_01` percentile distributions, closed-form linear neutralization, and AST bytecode generation.
  - *Validation (Ensemble & Fleet):* End-to-end inference verification across all 30 model strategies (25 Classic + 5 Signals).
  - *Integration (API & S3):* Blackbox validation of Numerai GraphQL API retry logic and S3 upload verification.
- **8.7 Control of Non-conforming Outputs:** Any NaN prediction, missing model weight, or collinear signal halts submission immediately.

---

### 7. Clause 9: Performance Evaluation
- **9.1 Monitoring & Measurement:** Automated 30-minute telemetry sweeps via `scripts/monitor_portfolio.py` tracking round status, fleet submission hashes, and NumerBay sales.
- **9.2 Internal Audit:** Automated QMS audit executed via [`scripts/audit_iso9001_compliance.py`](file:///Users/ishantpanchal/numerai-quant/scripts/audit_iso9001_compliance.py).

---

### 8. Clause 10: Continual Improvement
- **10.1 Non-conformity and Corrective Action:** Era drawdowns or factor decay trigger genetic formula re-synthesis and purged walk-forward cross-validation.
- **10.2 Continual Improvement Cycle:** Regular expansion to new targets (`target_cyrusd_60`, `target_agnes_60`) and multi-factor Signals generation.
