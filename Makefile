# Mac/Linux (and Windows with make + Git Bash). Windows without make: setup.ps1 / run.bat.
PY := .venv/bin/python
.PHONY: setup smoke run test clean
setup:
	./setup.sh
smoke:
	$(PY) -m mvp.smoke
run:
	./run.sh
test:
	$(PY) -m pytest -q tests
clean:
	rm -rf out_smoke out_demo .pytest_cache
