# M2 integrated manuscript and reproducible project

**Certifying Through Repair: Decoder Selection and Finite-Horizon Representation Intervention**

Author metadata is retained from the supplied manuscript: Zijie Huang, Independent Researcher.
This is an integrated research draft, not an accepted publication or an independently peer-reviewed correctness certificate.

The release combines the supplied 29-page fixed-confidence manuscript with the finite-horizon Bayes development. It contains full statements and proofs, generated tables and vector figures, executable implementations, tests, original-output comparisons, and two source distributions. Start with [the Chinese reading and reproduction guide](docs/阅读与复现说明.md).

## Quick start

Python 3.10 or later, NumPy and Matplotlib are needed for the complete computational pipeline. The exact Bayes core uses the standard library; the complete test suite also exercises NumPy-based certification code.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[figures]"
python scripts/reproduce.py
```

PDF generation additionally requires a TeX installation with pdfLaTeX, BibTeX and latexmk (including the packages imported by paper/main.tex). On Windows activate the virtual environment using its Scripts directory.

If TeX is unavailable:

```bash
python scripts/reproduce.py --skip-pdf
```

This explicitly skips PDF compilation; it does not establish that a fresh TeX build succeeded.

## Outputs and commands

| Purpose | Command / location |
|---|---|
| Complete verified pipeline | python scripts/reproduce.py |
| Tests | python scripts/run_tests.py |
| Final PDF | output/pdf/M2_Integrated_Manuscript.pdf |
| Stage report | results/validation_report.json |
| Integration checks | results/integration_report.json |
| Numerical outputs | results/*.csv and results/*.json |
| Publication figures and tables | paper/figures/ and paper/tables/ |
| Manuscript | paper/main.tex and paper/sections/ |
| Exact Bayes command line | python -m m2fh.cli configs/B0_original_bayes.json --policy gate --output results/cli_example.json |
| Visible-event replay example | python examples/replay_example.py |
| Build both release archives | python scripts/package_release.py |
| Compile manuscript alone | latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex |

The release archives are written next to this project directory. The arXiv tarball contains the TeX sources, bibliography, generated tables and PDF figures, so it compiles without Python. The complete ZIP also includes code, configurations, results and documentation. MANIFEST.json hashes all packaged files except itself.

## Scientific scope

There are two different contracts. Fixed-confidence certification declares the **old** representation's label, with a uniform error constraint. Finite-horizon Bayes control declares the **current** representation's label and minimizes expected task, action and terminal costs. They share an intervention viewpoint but are not interchangeable optimization problems.

Relative to the supplied manuscript, additions include classwise decoder factorization and a polynomial implementation of its exact race, a finite-deadline information inequality with a matching shrinking-post-channel limit, and the integrated Bayes gate/repair-region analysis. The scope and proof assumptions are listed in docs/定理与假设清单.md. Standard belief-state sufficiency, alpha-vector dynamic programming, Blackwell monotonicity and testing inequalities are explicitly attributed; they are not claimed as new.

The 29-page source archive included generated data but no original Python repository. The included implementation is a reconstruction and extension. It reproduces the supplied finite-protocol and Bayes sensitivity data up to floating-point error. See provenance/input_sources.json and docs/整合对应表.md.

These are controlled finite-model computations, not experiments on a deployed memory system, real LLM editing, or a learned representation. They do not validate physical intervention semantics or establish global novelty. No submission or external publication has been performed.

