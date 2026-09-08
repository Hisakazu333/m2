"""Publication figures: vector PDF and 220 dpi PNG, generated from saved CSV."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT/"results", ROOT/"paper/figures"
FIG.mkdir(exist_ok=True)
COLORS = {"KEEP": "#4477AA", "VERIFY": "#CC8844", "REPAIR": "#228877",
          "FALLBACK": "#8866AA"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False,
                     "axes.spines.right":False, "axes.labelsize":10, "legend.fontsize":8,
                     "pdf.fonttype":42, "ps.fonttype":42, "savefig.bbox":"tight"})


def read(name):
    with (DATA/name).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(FIG/(name+".pdf"))
    fig.savefig(FIG/(name+".png"), dpi=220)
    plt.close(fig)


def main():
    rows = [r for r in read("regions.csv") if r["horizon"] == "1"]
    p = np.array([float(r["prior"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.5,3.25))
    for a in ("KEEP", "VERIFY", "REPAIR"):
        ax.plot(p, [float(r[a]) for r in rows], label=a, lw=1.8, color=COLORS[a])
    for lo, hi in ((3/22,1/3),(7/11,1)):
        ax.axvspan(lo, hi, color=COLORS["REPAIR"], alpha=.10)
    ax.axvline(.5, color="#666666", ls=":", lw=1.1)
    ax.set(xlabel=r"Prior inadequacy probability $p$", ylabel="One-round Bayes loss",
           xlim=(0,1), ylim=(0,2.15))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5,1.16))
    ax.text(.53,1.77,"Declaration boundary",fontsize=8,color="#555555")
    save(fig,"disconnected_region")

    rows = read("regions.csv")
    horizons = sorted(set(int(r["horizon"]) for r in rows))
    names = ["KEEP","VERIFY","REPAIR"]
    image = np.array([[names.index(r["action"]) for r in rows if int(r["horizon"]) == h]
                      for h in horizons])
    fig, ax = plt.subplots(figsize=(6.5,2.9))
    cmap = ListedColormap([COLORS[n] for n in names])
    ax.imshow(image,origin="lower",aspect="auto",interpolation="nearest",
              extent=(-.005,1.005,-.5,len(horizons)-.5),cmap=cmap,
              norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
    ax.set_yticks(range(len(horizons)), labels=horizons)
    ax.axvline(.5,color="white",ls=":",lw=1.3)
    ax.set(xlabel=r"Prior inadequacy probability $p$",ylabel="Remaining rounds")
    ax.legend(handles=[Patch(color=COLORS[n],label=n) for n in names],
              frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.20))
    save(fig,"horizon_regions")

    rows = read("no_information_gap.csv")
    fig, ax = plt.subplots(figsize=(6.5,3))
    h = [int(r["horizon"]) for r in rows]
    ax.plot(h,[float(r["gate"]) for r in rows],label="Declaration gate",color=COLORS["VERIFY"])
    ax.plot(h,[float(r["joint"]) for r in rows],label="Joint optimum",color=COLORS["REPAIR"])
    ax.fill_between(h,[float(r["joint"]) for r in rows],[float(r["gate"]) for r in rows],
                    color=COLORS["VERIFY"],alpha=.12)
    ax.set(xlabel=r"Horizon $T$",ylabel="Exact Bayes loss")
    ax.legend(frameon=False)
    save(fig,"no_information_gap")

    rows = read("blackwell.csv")
    fig, ax = plt.subplots(figsize=(6.5,3))
    for h, color in zip((1,3,5,8),("#4477AA","#228877","#CC8844","#8866AA")):
        part = [r for r in rows if int(r["horizon"]) == h]
        ax.plot([float(r["error"]) for r in part],[float(r["value"]) for r in part],
                marker="o",ms=3,label=f"T={h}",color=color)
    ax.set(xlabel="Verification crossover probability",ylabel="Optimal Bayes loss")
    ax.legend(frameon=False,ncol=4)
    save(fig,"blackwell")

    rows = read("post_information.csv")
    fig, ax = plt.subplots(figsize=(6.5,3))
    for e, color in zip((.5,.2,.05),("#4477AA","#228877","#8866AA")):
        part=[r for r in rows if float(r["post_error"]) == e]
        ax.plot([float(r["prior"]) for r in part],[float(r["value"]) for r in part],
                label=f"Post-repair error={e:g}",color=color)
    ax.axvline(.5,color="#777777",ls=":",lw=1)
    ax.set(xlabel=r"Prior inadequacy probability $p$",ylabel="Optimal Bayes loss")
    ax.legend(frameon=False)
    save(fig,"post_information")

    rows = read("informative_gate_upper_bound.csv")
    fig, axes = plt.subplots(1,2,figsize=(6.5,2.8))
    h = [int(r["horizon"]) for r in rows]
    axes[0].semilogx(h,[float(r["gate_gap_upper_bound"]) for r in rows],
                    marker="o",ms=3,color=COLORS["VERIFY"])
    axes[0].set(xlabel=r"Horizon $T$",ylabel="Upper bound on gate gap")
    axes[1].semilogx(h,[int(r["best_tested_prefix"]) for r in rows],
                    marker="o",ms=3,color=COLORS["KEEP"])
    axes[1].set(xlabel=r"Horizon $T$",ylabel="Best tested prefix length")
    fig.tight_layout()
    save(fig,"informative_gate_bound")
    print("Six figures generated in PDF and PNG")


if __name__ == "__main__":
    main()
