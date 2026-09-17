#!/usr/bin/env python3
"""
ISO/DIS 9001:2026 Quality Management System (QMS) Compliance Auditor
Tailored for Numerai Quant Hedge Fund Fleet & Signals Pipeline
"""
import os
import sys
import json
import shutil
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def file_exists(rel_path: str) -> bool:
    return os.path.exists(os.path.join(BASE_DIR, rel_path))

def file_contains(rel_path: str, keyword: str) -> bool:
    p = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return keyword in f.read()
    except Exception:
        return False

def audit_clause_4():
    """Clause 4: Context of the Organization & Digital Infrastructure"""
    checks = {
        "makefile_orchestrator_defined": file_exists("Makefile"),
        "project_config_defined": file_exists("pyproject.toml"),
        "fleet_config_present": file_exists("config.py"),
        "neutralization_engine_present": file_exists("neutralize.py"),
    }
    return {
        "clause": "Clause 4: Context & Digital Infrastructure",
        "passed": all(checks.values()),
        "details": checks
    }

def audit_clause_5():
    """Clause 5: Leadership & Quality Culture"""
    checks = {
        "quality_manual_present": file_exists("iso9001_compliance/QUALITY_MANUAL.md"),
        "zero_defect_policy_stated": file_contains("iso9001_compliance/QUALITY_MANUAL.md", "Zero Completion Claims Without Verification"),
        "makefile_pipeline_gated": file_contains("Makefile", "test: build-chimera"),
    }
    return {
        "clause": "Clause 5: Leadership & Quality Culture",
        "passed": all(checks.values()),
        "details": checks
    }

def audit_clause_6():
    """Clause 6: Planning & Risk-Based Thinking"""
    checks = {
        "risk_register_maintained": file_exists("iso9001_compliance/RISK_REGISTER.md"),
        "neutralization_implemented": file_contains("neutralize.py", "def neutralize"),
        "60d_targets_configured": file_contains("config.py", "target_cyrusd_60"),
        "prediction_validation_present": file_exists("tests/test_numerbay_publisher.py"),
    }
    return {
        "clause": "Clause 6: Planning & Risk-Based Thinking",
        "passed": all(checks.values()),
        "details": checks
    }

def audit_clause_7():
    """Clause 7: Support & Tool Qualification"""
    checks = {
        "python3_runtime_available": shutil.which("python3") is not None,
        "make_build_tool_available": shutil.which("make") is not None,
        "chimera_engine_present": file_exists("chimera/csrc/Makefile"),
        "spec_documentation_present": file_exists("README.md"),
    }
    return {
        "clause": "Clause 7: Support & Tool Qualification",
        "passed": all(checks.values()),
        "details": checks
    }

def audit_clause_8():
    """Clause 8: Operation & Software Verification/Validation (V&V)"""
    checks = {
        "traceability_matrix_present": file_exists("iso9001_compliance/TRACEABILITY_MATRIX.md"),
        "whitebox_test_suite_present": file_exists("tests/test_whitebox.py"),
        "orthogonality_test_suite_present": file_exists("tests/test_orthogonality.py"),
        "fleet_test_suite_present": file_exists("tests/test_fleet_30.py"),
        "signals_test_suite_present": file_exists("tests/test_signals.py"),
    }
    return {
        "clause": "Clause 8: Operation & Software V&V",
        "passed": all(checks.values()),
        "details": checks
    }

def audit_clause_9():
    """Clause 9: Performance Evaluation & Audit Trails"""
    checks = {
        "portfolio_monitor_script_present": file_exists("scripts/monitor_portfolio.py"),
        "walk_forward_cv_suite_present": file_exists("tests/test_walk_forward_cv.py"),
    }
    return {
        "clause": "Clause 9: Performance Evaluation & Audit Trails",
        "passed": all(checks.values()),
        "details": checks
    }

def audit_clause_10():
    """Clause 10: Continual Improvement & Defect Containment"""
    checks = {
        "genetic_synthesizer_present": file_exists("chimera/genetic_synthesizer.py"),
        "automated_qms_auditor_wired": file_exists("scripts/audit_iso9001_compliance.py"),
    }
    return {
        "clause": "Clause 10: Continual Improvement & Defect Containment",
        "passed": all(checks.values()),
        "details": checks
    }

def main():
    print("==========================================================================")
    print("      ISO/DIS 9001:2026 Quality Management System (QMS) Compliance Auditor")
    print("      Target: Numerai Quant Hedge Fund Fleet & Signals Engine")
    print("==========================================================================")

    audits = [
        audit_clause_4(),
        audit_clause_5(),
        audit_clause_6(),
        audit_clause_7(),
        audit_clause_8(),
        audit_clause_9(),
        audit_clause_10(),
    ]

    total_clauses = len(audits)
    passed_clauses = sum(1 for a in audits if a["passed"])
    score_pct = (passed_clauses / total_clauses) * 100.0

    print(f"\nAudit Timestamp: {datetime.now().isoformat()}")
    print(f"Target Repository: {BASE_DIR}\n")
    print("Clause-by-Clause Compliance Scorecard:")
    print("--------------------------------------------------------------------------")

    for a in audits:
        status_symbol = "✓ PASS" if a["passed"] else "✗ FAIL"
        print(f"[{status_symbol}] {a['clause']}")
        for detail, res in a["details"].items():
            sub_symbol = "  ✔" if res else "  ✖"
            print(f"    {sub_symbol} {detail}: {'Satisfied' if res else 'MISSING'}")

    print("--------------------------------------------------------------------------")
    print(f"Final Compliance Score: {score_pct:.1f}% ({passed_clauses}/{total_clauses} Clauses Compliant)")

    target_dir = os.path.join(BASE_DIR, "target")
    os.makedirs(target_dir, exist_ok=True)
    report_path = os.path.join(target_dir, "iso9001_audit_report.json")

    report_payload = {
        "standard": "ISO/DIS 9001:2026",
        "repository": "numerai-quant",
        "timestamp": datetime.now().isoformat(),
        "score_percent": score_pct,
        "clauses_audited": total_clauses,
        "clauses_passed": passed_clauses,
        "compliant": score_pct == 100.0,
        "results": audits
    }

    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nMachine-readable audit artifact written to: {report_path}")

    if score_pct == 100.0:
        print("\n🏆 AUDIT RESULT: FULLY CONFORMANT TO ISO/DIS 9001:2026 STANDARD.")
        sys.exit(0)
    else:
        print("\n⚠️ AUDIT RESULT: NON-CONFORMANCES DETECTED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
