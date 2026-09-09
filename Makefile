.PHONY: all test demo lint clean

all: test

test:
	./venv/bin/python -m pytest -v

demo:
	./venv/bin/python -m pytest tests/test_whitebox.py -v
	@echo "=== Numerai Quant: 12/12 Passing with Closed-Form Linear Neutralization ==="

lint:
	/Users/ishantpanchal/.local/bin/ruff check .

clean:
	rm -rf __pycache__ .pytest_cache
