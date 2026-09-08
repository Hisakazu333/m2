"""Vector figures for certification, shrinking channels and original fixture."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from make_figures import plt, np, read, save, COLORS


def main():
    rows=json.loads((ROOT/"results/certification_coefficients.json").read_text())
    fig,ax=plt.subplots(figsize=(6.4,3.0))
    x=np.arange(len(rows))
    ax.bar(x-.18,[r["joint_coefficient"] for r in rows],width=.36,
           color=COLORS["REPAIR"],label="Joint decoder")
    ax.bar(x+.18,[r["separated_coefficient"] for r in rows],width=.36,
           color=COLORS["VERIFY"],label="Certify first")
    ax.set_xticks(x,["$E_1$","$E_2$","$E_3$"])
    ax.set(ylabel=r"Cost coefficient per $\log(1/\delta)$")
    ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.2))
    save(fig,"certification_coefficients")

    rows=[r for r in read("shrinking_channel_bound.csv") if r["environment"]=="1"]
    fig,ax=plt.subplots(figsize=(6.4,3.1))
    ax.semilogx([float(r["log_inverse_delta"]) for r in rows],
                [float(r["normalized_lower_bound"]) for r in rows],
                marker="o",ms=3,color=COLORS["REPAIR"],label="Finite lower bound / L")
    ax.axhline(float(rows[0]["separated_coefficient"]),color=COLORS["VERIFY"],
               ls="--",lw=1.3,label="Separated limit")
    ax.axhline(0,color=COLORS["KEEP"],ls=":",lw=1.3,label="Frozen-channel limit")
    ax.set(xlabel=r"$L=\log(1/\delta)$",ylabel="Normalized certification cost",ylim=(-2,57))
    ax.legend(frameon=False,loc="center right")
    save(fig,"shrinking_channel")

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
    gates=read("gate_threshold_sweep.csv")
    axes[0].step([float(r["threshold"]) for r in gates],[float(r["gate"]) for r in gates],
                 where="post",color=COLORS["VERIFY"],label="Repair-only gate")
    axes[0].axhline(float(gates[0]["joint"]),color=COLORS["REPAIR"],label="Joint")
    for p in (.5,.95):
        axes[0].axvline(p,color="#888888",lw=.8,ls=":")
    axes[0].set(xlabel="Gate threshold",ylabel="Bayes loss",xlim=(0,1))
    axes[0].legend(frameon=False,loc="lower right")
    rows=[r for r in read("original_sensitivity.csv") if r["parameter"]=="repair_success"]
    for key,label,color in (("joint","Joint",COLORS["REPAIR"]),
                            ("gate","Gate 0.95",COLORS["VERIFY"]),
                            ("no_repair","No repair",COLORS["KEEP"])):
        axes[1].plot([float(r["setting"]) for r in rows],[float(r[key]) for r in rows],
                     label=label,color=color,marker="o",ms=3,
                     ls="--" if key=="no_repair" else "-")
    axes[1].set(xlabel="Repair success probability",ylabel="Bayes loss")
    axes[1].legend(frameon=False,loc="lower left")
    fig.tight_layout()
    save(fig,"original_sensitivity")
    print("Three integrated vector figures generated")


if __name__=="__main__":
    main()
