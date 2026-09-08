"""Generate every manuscript result from frozen finite configurations."""
from pathlib import Path
from fractions import Fraction as F
import csv
import hashlib
import json
import math
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from m2fh.model import load_model
from m2fh.families import binary_model
from m2fh.solver import Solver
from m2fh.enumerator import exhaustive_value
from m2fh.policies import baseline, evaluate, diagnostics, oracle_value, advantage_sum
from m2fh.theory import no_information_gap, repair_threshold

OUT = ROOT/"results"
OUT.mkdir(exist_ok=True)


def save_csv(name, rows):
    with (OUT/name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hash():
    hasher = hashlib.sha256()
    for directory in ("src", "scripts", "configs", "tests"):
        for path in sorted((ROOT/directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                hasher.update(str(path.relative_to(ROOT)).encode())
                hasher.update(path.read_bytes())
    return hasher.hexdigest()


def run():
    start = time.perf_counter()
    benchmark, diagnostics_all = [], {}
    names = ("joint", "gate", "myopic", "greedy_voi", "verify_then_act",
             "never_repair", "always_repair")
    for config in sorted((ROOT/"configs").glob("F*.json")):
        model = load_model(config)
        row = {"instance": config.stem.split("_")[0], "horizon": model.horizon,
               "model_sha256": model.digest(), "oracle": float(oracle_value(model))}
        diagnostics_all[row["instance"]] = {}
        joint = Solver(model).value()
        for name in names:
            choose = baseline(model, name)
            result = diagnostics(model, choose)
            value = result["loss"]
            assert value == evaluate(model, choose)
            if name in ("joint", "gate"):
                assert value - joint == advantage_sum(model, choose)
            row[name] = float(value)
            row[name+"_exact"] = str(value)
            diagnostics_all[row["instance"]][name] = result
        gap = F(row["gate_exact"])-F(row["joint_exact"])
        row["gate_gap"] = float(gap)
        row["gate_gap_exact"] = str(gap)
        benchmark.append(row)
        print(f"{row['instance']}: joint={row['joint']:.8f}, gate={row['gate']:.8f}", flush=True)
    save_csv("benchmark.csv", benchmark)
    (OUT/"diagnostics.json").write_text(json.dumps(diagnostics_all, indent=2, default=str))

    check_rows = []
    for h in range(4):
        for p in ("0", "1/5", "1/2", "4/5", "1"):
            model = binary_model(horizon=h, prior=p)
            exact, count = exhaustive_value(model)
            dp = Solver(model).value()
            assert dp == exact
            check_rows.append({"horizon": h, "prior": p, "trees": count,
                               "value_exact": str(exact), "difference_exact": str(dp-exact)})
    save_csv("enumeration_checks.csv", check_rows)
    print("Independent exhaustive checks complete", flush=True)

    regions = []
    for h in (1, 2, 3, 4, 6, 8):
        model = binary_model(horizon=h)
        sol = Solver(model)
        for k in range(101):
            p = F(k, 100)
            b = (1-p, p)
            v = sol.value(h, b, "old")
            qs = sol.qvalues[h, b, "old"]
            regions.append({"horizon": h, "prior": float(p), "action": sol.action(h, b, "old"),
                            "repair_optimal": int(qs["REPAIR"] == v),
                            "KEEP": float(qs["KEEP"]), "VERIFY": float(qs["VERIFY"]),
                            "REPAIR": float(qs["REPAIR"]), "value": float(v)})
        print(f"Action regions h={h} complete", flush=True)
    save_csv("regions.csv", regions)

    post_rows = []
    for error in ("1/2", "1/5", "1/20"):
        model = binary_model(horizon=5, post_error=error)
        sol = Solver(model)
        for k in range(101):
            p = F(k, 100)
            b = (1-p, p)
            val = sol.value(5, b, "old")
            post_rows.append({"post_error": float(F(error)), "prior": float(p),
                              "value": float(val), "action": sol.action(5, b, "old"),
                              "repair_value": float(sol.qvalues[5, b, "old"]["REPAIR"])})
        print(f"Post-repair channel error={error} complete", flush=True)
    save_csv("post_information.csv", post_rows)

    info_rows = []
    for h in (1, 3, 5, 8):
        for k in range(11):
            error = F(k, 20)
            model = binary_model(horizon=h, verify_error=str(error), verify_task="old")
            info_rows.append({"horizon": h, "error": float(error),
                              "value": float(Solver(model).value())})
    save_csv("blackwell.csv", info_rows)

    gap_rows = []
    for h in range(1, 51):
        model = binary_model(horizon=h, r="2", s="1/2", repair_cost="1",
                             terminal_penalty="1", verify=False)
        joint, gate = Solver(model).value(), Solver(model, "gate").value()
        formula = no_information_gap(h, "3/10", "2", "1/2", "1")
        assert gate-joint == formula
        gap_rows.append({"horizon": h, "joint": float(joint), "gate": float(gate),
                         "gap": float(formula), "threshold": float(repair_threshold(h, 2, "1/2", 1))})
    save_csv("no_information_gap.csv", gap_rows)

    # Fixed-prefix gate: use exact rational probabilities, then convert to float.
    # VERIFY incurs OLD loss plus c_v. The policy uses the Bayes 1/2 gate.
    err_cache = {}
    p, e = F("3/10"), F("1/5")
    for n in range(101):
        err0 = err1 = F(0)
        for k in range(n+1):
            q0 = math.comb(n, k)*e**k*(1-e)**(n-k)
            q1 = math.comb(n, k)*(1-e)**k*e**(n-k)
            if p*q1 >= (1-p)*q0:
                err0 += q0
            else:
                err1 += q1
        err_cache[n] = (err0, err1)
    prefix_rows = []
    for h in (5, 10, 20, 50, 100, 200, 500, 1000):
        candidates = []
        for n in range(min(100, h-1)+1):
            err0, err1 = err_cache[n]
            value = (n*(p*2+F("1/4")) + (1-p)*err0*(1+(h-n)*F("1/2"))
                     + p*(1-err1) + p*err1*(h-n)*2 + ((1-p)*err0+p*err1))
            candidates.append((value, n))
        value, n = min(candidates)
        prefix_rows.append({"horizon": h, "best_tested_prefix": n, "policy_risk": float(value),
                            "oracle": float(p), "gate_gap_upper_bound": float(value-p),
                            "policy_risk_exact": str(value), "max_prefix_tested": min(100, h-1)})
    save_csv("informative_gate_upper_bound.csv", prefix_rows)

    scaling = []
    for h in (1, 2, 3, 4, 6, 8, 10, 12):
        model = binary_model(horizon=h)
        t0 = time.perf_counter()
        sol = Solver(model)
        value = sol.value()
        scaling.append({"horizon": h, "cached_beliefs": len(sol.values),
                        "seconds": time.perf_counter()-t0, "value": float(value)})
    save_csv("scaling.csv", scaling)

    model = load_model(ROOT/"configs/F5_failed_repair.json")
    (OUT/"optimal_policy.json").write_text(
        json.dumps(Solver(model).export_policy(), indent=2))
    (OUT/"frozen_model_F5.json").write_text(json.dumps(model.to_dict(), indent=2))
    metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "arithmetic": "fractions.Fraction for all solver values and posterior updates",
        "experiment_type": "deterministic exact finite-model computations",
        "random_seed": None, "seed_reason": "No Monte Carlo sampling is used",
        "source_sha256": source_hash(), "external_git_commit": None,
        "elapsed_seconds": time.perf_counter()-start,
        "enumeration_cases": len(check_rows),
        "largest_enumerated_policy_count": max(r["trees"] for r in check_rows),
        "disconnected_region": ["[3/22, 1/3]", "[7/11, 1]"],
        "analytic_no_information_gap_at_T10": "3/2",
        "baseline_names": list(names),
    }
    (OUT/"metadata.json").write_text(json.dumps(metadata, indent=2))
    make_tables(benchmark, metadata)
    print(f"Complete in {metadata['elapsed_seconds']:.2f}s", flush=True)


def make_tables(rows, metadata):
    table = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Instance & Oracle & Joint & Gate & Myopic & Gate gap\\", r"\midrule"]
    for row in rows:
        table.append(f"{row['instance']} & {row['oracle']:.4f} & {row['joint']:.4f} & "
                     f"{row['gate']:.4f} & {row['myopic']:.4f} & {row['gate_gap']:.4f}"+r"\\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (ROOT/"paper/tables/benchmark.tex").write_text("\n".join(table)+"\n")
    (ROOT/"paper/tables/run_macros.tex").write_text(
        "\\newcommand{\\EnumerationCases}{"+str(metadata["enumeration_cases"])+"}\n"
        "\\newcommand{\\LargestTreeCount}{"+str(metadata["largest_enumerated_policy_count"])+"}\n")


if __name__ == "__main__":
    run()
