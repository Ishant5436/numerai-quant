import os
import subprocess
import sys

def test_ast_safety_invariants_mechanical_audit():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(base_dir, "scripts", "audit_safety_invariants.py")
    assert os.path.exists(script), f"Audit script missing: {script}"

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    assert result.returncode == 0, f"Safety invariant audit failed:\n{result.stdout}\n{result.stderr}"
    assert "[PASS]" in result.stdout
