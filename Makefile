.PHONY: all test demo lint clean

all: test

test:
	./venv/bin/python -m pytest -v

demo:
	./venv/bin/python -m pytest tests/test_whitebox.py -v
	@echo "=== Numerai Quant: 12/12 Passing with Closed-Form Linear Neutralization ==="

lint:
	uv run ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
