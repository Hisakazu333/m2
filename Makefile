.PHONY: reproduce test experiments figures paper package
PYTHON ?= python3
reproduce:
	$(PYTHON) scripts/reproduce.py
test:
	$(PYTHON) scripts/run_tests.py
experiments:
	$(PYTHON) scripts/run_integrated_experiments.py
	$(PYTHON) scripts/run_experiments.py
figures:
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/make_integrated_figures.py
paper:
	latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
package:
	$(PYTHON) scripts/package_release.py
