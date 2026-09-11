# Anonymization for the TMLR double-blind submission

This note documents the anonymization applied for the double-blind TMLR
submission of "Certifying Through Repair: Decoder Selection and
Finite-Horizon Representation Intervention" and how to use the anonymized
supplementary code package.

## 1. PDF metadata fix (double-blind vulnerability)

`paper/main_tmlr.tex` previously set `pdfauthor={Zijie Huang}` in
`\hypersetup`, which wrote the author name into the PDF document metadata
(`pdfinfo` showed `Author: Zijie Huang`) even though the rendered first page
was anonymous. This is a double-blind leak. It is now
`pdfauthor={Anonymous}`; `pdftitle` is unchanged. `main_preprint.tex`
retains full attribution and is not part of the double-blind submission.

Verification of the rebuilt `m2_tmlr_submission.pdf`:

- `pdfinfo` reports `Author: Anonymous` (no personal name anywhere in the
  metadata), `pdftitle` preserved.
- First page shows "Anonymous authors / Paper under double-blind review"
  and the header "Under review as submission to TMLR".
- Compiled with tectonic, 0 errors.

The bibliography keeps the third-person self-citation, which is standard
TMLR double-blind practice; the leak was in the metadata, not the
reference list.

## 2. Anonymized supplementary code package

The TMLR supplementary package is built from this repository's code assets
(`src/m2fh/`, `tests/`, `scripts/`, `examples/`, `configs/`,
`requirements-figures.txt`, `pyproject.toml`, `Makefile`, `results/`,
`provenance/`, `schemas/`) with the following removals/scrubs, each verified
by grep over the packaged tree (0 remaining matches):

| Item | Handling |
|---|---|
| Author name (Zijie Huang / Hisakazu / hisakazu / Huang, Zijie) | removed; README rewritten as "Anonymous authors" |
| Emails (huangzijie@cuyitech.com, 750691178@qq.com) | removed |
| GitHub user / repository links (github.com/Hisakazu333) | removed |
| Local paths (/Users/hisakazu/, /home/<user>) | none present; verified |
| arXiv:2608.02267 (companion-preprint identifier) | not present in code assets; the stale citation key `huang2026` in `results/logs/*.txt` was generalized to `companion2026` |
| `.git` directory | excluded from the package |
| `README.md` | rewritten anonymously; run instructions kept |
| `MANIFEST.json` | regenerated (SHA-256 over the anonymized file set, excluding itself) |

The package intentionally excludes `paper/` and `output/` (the manuscript
sources carry attribution and are delivered separately).

## 3. Using the supplementary package

```bash
unzip supplementary_code.zip
cd supplementary_code
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[figures]"
python scripts/reproduce.py --skip-pdf   # all computational stages
python scripts/run_tests.py              # 32 tests, all passing
```

`--skip-pdf` is required because the manuscript sources are not part of the
supplementary package. Anonymization was verified not to change behavior:
after scrubbing, a fresh `pip install -e .` plus `python scripts/run_tests.py`
passes all 32 tests.
