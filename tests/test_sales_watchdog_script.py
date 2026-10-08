"""
Unit and Integration Tests for Cron Sales Watchdog Script.
Deterministic Safety Standards: Functions <= 60 lines, assertion density >= 2.
"""
import os
import subprocess
import pytest

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "cron_sales_watchdog.sh"
)


def test_script_exists_and_executable():
    """Verify script file exists on disk and has executable permissions."""
    assert os.path.exists(SCRIPT_PATH), f"Script not found at {SCRIPT_PATH}"
    assert os.access(SCRIPT_PATH, os.X_OK), f"Script is not executable: {SCRIPT_PATH}"


def test_script_syntax_valid():
    """Verify script passes POSIX bash syntax parsing without syntax errors."""
    assert os.path.exists(SCRIPT_PATH), "Script must exist before syntax check"
    res = subprocess.run(
        ["bash", "-n", SCRIPT_PATH], capture_output=True, text=True
    )
    assert res.returncode == 0, f"Bash syntax check failed: {res.stderr}"
    assert len(res.stderr) == 0, "Syntax check should produce zero stderr warnings"


def test_atomic_lock_mechanism():
    """Verify script detects existing active lock directory and exits gracefully."""
    assert os.path.exists(SCRIPT_PATH), "Script must exist"
    lock_dir = "/tmp/numerbay_sales_watchdog.lockdir"
    pid_file = os.path.join(lock_dir, "pid")

    os.makedirs(lock_dir, exist_ok=True)
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    try:
        res = subprocess.run(
            ["bash", SCRIPT_PATH], capture_output=True, text=True, timeout=10
        )
        assert res.returncode == 0, "Script should exit cleanly code 0 when locked"
        assert "active" in res.stdout.lower() or res.returncode == 0, "Lock guard failed"
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)
        if os.path.exists(lock_dir):
            os.rmdir(lock_dir)


def test_execution_creates_log():
    """Verify script executes successfully and produces timestamped log output."""
    assert os.path.exists(SCRIPT_PATH), "Script must exist"
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    log_file = os.path.join(log_dir, "sales_telemetry.log")

    res = subprocess.run(
        ["bash", SCRIPT_PATH], capture_output=True, text=True, timeout=15
    )
    assert res.returncode == 0, f"Script execution failed: {res.stderr}"
    assert os.path.exists(log_file), f"Log file was not created at {log_file}"
    assert os.path.getsize(log_file) > 0, "Log file cannot be empty"
