# Figure revision note (September 2026)

All ten figures in the paper were redrawn to publication grade (vector PDFs,
muted colorblind-safe palette, consistent typography, leader-line annotations).

**No numerical content changed.** Every plotted value is read from the
checked-in artifacts under `results/` (the same CSV/JSON files as before);
each redraw script asserts equality with those artifacts before writing its
figure. A new overview figure `figures/fig_contracts.pdf` was added at the
start of Section 2 (the two contracts over one intervention model) and is
referenced from the Introduction.

The compiled PDFs in the release package were regenerated with the new
figures. Reproduction scripts for the figures accompany the release.
