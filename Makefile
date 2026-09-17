.PHONY: all test build-chimera demo lint audit-iso9001 clean

all: test

build-chimera:
	$(MAKE) -C chimera/csrc

test: build-chimera
	./venv/bin/python -m pytest -v

audit-iso9001:
	@echo "=== Verifying Numerai Quant Against ISO/DIS 9001:2026 Standards ==="
	python3 scripts/audit_iso9001_compliance.py

demo:
	./venv/bin/python -m pytest tests/test_whitebox.py -v
	@echo "=== Numerai Quant: 12/12 Passing with Closed-Form Linear Neutralization ==="

lint:
	uv run ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache target/
