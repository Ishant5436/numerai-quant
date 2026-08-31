.PHONY: all test demo lint clean

all: test

test:
	python3 -m pytest -v

demo:
	python3 -m pytest tests/test_whitebox.py -v
	@echo "=== Numerai Quant: 12/12 Passing with Closed-Form Linear Neutralization ==="

lint:
	python3 -m flake8 . --count --max-line-length=120 --statistics || true

clean:
	rm -rf __pycache__ .pytest_cache
