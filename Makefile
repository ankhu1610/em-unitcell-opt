.PHONY: setup test snapshot check-snapshot
PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[test]"

test:
	$(PYTHON) -m pytest tests/ -v

snapshot:
	$(PYTHON) scripts/build_repository_snapshot.py

check-snapshot:
	$(PYTHON) scripts/build_repository_snapshot.py --check
