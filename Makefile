.PHONY: setup test aggregate baselines priors pinn pwe-verify calibrate multifidelity inverse-test benchmark figures reproduce

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

aggregate:
	$(PYTHON) scripts/aggregate.py

baselines:
	$(PYTHON) scripts/run_baselines.py

priors:
	$(PYTHON) scripts/check_physics_priors.py

pinn:
	$(PYTHON) scripts/train_pinn.py

pwe-verify:
	$(PYTHON) scripts/verify_pwe.py

calibrate:
	$(PYTHON) scripts/calibrate_pwe.py

multifidelity:
	$(PYTHON) scripts/train_multifidelity.py

inverse-test:
	$(PYTHON) scripts/test_inverse.py

benchmark:
	$(PYTHON) scripts/run_benchmarks.py

figures:
	$(PYTHON) scripts/generate_figures.py

reproduce: test baselines priors pinn pwe-verify calibrate multifidelity inverse-test benchmark figures
	@echo "All reproducibility steps completed on CPU."
