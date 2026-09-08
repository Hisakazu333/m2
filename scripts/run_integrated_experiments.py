"""All certification, deadline, original-fixture and integration results."""
from pathlib import Path
from fractions import Fraction as F
import csv
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from m2fh.certification import RepairExperiment, finite_protocol
from m2fh.decoder_structure import factorized_coefficients, decoder_from_scores
from m2fh.deadline import old_label_cost_lower_bound, shrinking_post_instance
from m2fh.pdf_fixture import uploaded_bayes_fixture, RepairOnlyGate
from m2fh.solver import Solver
from m2fh.model import dot
from m2fh.enumerator import enumerate_trees
from m2fh.policies import diagnostics
from m2fh.durations import expand_durations
from m2fh.families import binary_model
from run_experiments import save_csv
from audit_uploaded_pdf import conditional_errors


def write_table(filename, headers, rows, alignment):
    lines = [r"\begin{tabular}{"+alignment+"}",r"\toprule",
             " & ".join(headers)+r"\\",r"\midrule"]
    lines.extend(" & ".join(row)+r"\\" for row in rows)
    lines += [r"\bottomrule",r"\end{tabular}"]
    (ROOT/"paper/tables"/filename).write_text("\n".join(lines)+"\n")


def run():
    started = time.perf_counter()
    raw = json.loads((ROOT/"configs/certification_default.json").read_text())
    model_keys = ("pre","post","labels","costs","repair_success","repair_cost")
    args = {key:tuple(raw[key]) if isinstance(raw[key],list) else raw[key] for key in model_keys}
    x = RepairExperiment(**args)
    coefficients = factorized_coefficients(x)
    assert coefficients == x.coefficients()
    (ROOT/"results/certification_coefficients.json").write_text(json.dumps(coefficients,indent=2))
    write_table("analytic_table.tex",
        ["Environment",r"$D_i$",r"$g_i/D_i$",r"$g_i/D_i^{\rm sep}$","Decoder"],
        [[f"$E_{r['environment']}$",f"{r['rate']:.6f}",f"{r['joint_coefficient']:.6f}",
          f"{r['separated_coefficient']:.6f}","$("+",".join(map(str,r["decoder"]))+")$"]
         for r in coefficients],"lrrrr")
    budget = {key:raw[key] for key in ("delta","horizon","pre_limit","repair_cap")}
    finite = finite_protocol(x,**budget)+finite_protocol(x,**budget,separated=True)
    assert all(r["error"] < raw["delta"] and r["conservative_error_bound"] < raw["delta"] for r in finite)
    save_csv("certification_finite.csv",finite)
    write_table("finite_table.tex",
        ["Policy","Env.","Cost",r"$\mathbb E N_P$","Error","Timeout"],
        [[r["policy"].replace("separated","Certify first").replace("joint","Joint"),
          f"$E_{r['environment']}$",f"{r['cost']:.6f}",f"{r['pre_samples']:.6f}",
          f"{r['error']:.2e}",f"{r['timeout']:.2e}"] for r in finite],"llrrrr")
    print("Certification coefficients and fully charged finite protocol complete",flush=True)

    fixture = uploaded_bayes_fixture()
    (ROOT/"configs/B0_original_bayes.json").write_text(json.dumps(fixture.to_dict(),indent=2))
    values = []
    for name, solver in (("joint",Solver(fixture)),("gate_095",RepairOnlyGate(fixture)),
                         ("no_repair",Solver(fixture,"never_repair"))):
        d = diagnostics(fixture,solver.action)
        row = {"policy":name,"value":float(solver.value()),"exact_value":str(solver.value()),
               "action":solver.action(fixture.horizon,fixture.prior,fixture.initial_state),
               "conditional_errors":conditional_errors(fixture,solver)}
        values.append(row)
    (ROOT/"results/original_bayes.json").write_text(json.dumps(values,indent=2))
    write_table("original_bayes_table.tex",
        ["Policy","Bayes cost","First action",r"$e_1$",r"$e_2$",r"$e_3$"],
        [[r["policy"].replace("_",r"\_"),f"{r['value']:.8f}",r["action"].lower()]
         +[f"{e:.6f}" for e in r["conditional_errors"]] for r in values],"lllr rr".replace(" ",""))

    sensitivity = []
    grids = {
        "repair_cost":["0","1/10","1/5","2/5","4/5"],
        "repair_success":["0","1/5","1/2","4/5","1"],
        "verify_accuracy":["1/2","13/20","4/5","19/20","1"],
        "horizon":[1,2,4,6,8],
    }
    for key, settings in grids.items():
        for setting in settings:
            model = uploaded_bayes_fixture(**{key:setting})
            j,g,n = Solver(model),RepairOnlyGate(model),Solver(model,"never_repair")
            vj,vg,vn = j.value(),g.value(),n.value()
            assert vj <= vg and vj <= vn
            sensitivity.append({"parameter":key,"setting":float(F(setting)),
                "joint":float(vj),"gate":float(vg),"no_repair":float(vn),
                "joint_exact":str(vj),"gate_exact":str(vg),"no_repair_exact":str(vn),
                "action":j.action(model.horizon,model.prior,model.initial_state),
                "model_sha256":model.digest()})
        print(f"Original-fixture sensitivity {key} complete",flush=True)
    save_csv("original_sensitivity.csv",sensitivity)
    write_table("sensitivity_table.tex",["Parameter","Setting","Joint","Gate","No repair","Action"],
        [[r["parameter"].replace("_",r"\_"),f"{r['setting']:g}",f"{r['joint']:.6f}",
          f"{r['gate']:.6f}",f"{r['no_repair']:.6f}",r["action"].lower()]
         for r in sensitivity],"lrrrrl")

    gates = []
    for k in [0,25,40,45]+list(range(50,100)):
        model = uploaded_bayes_fixture(gate_threshold=str(F(k,100)))
        joint,gate = Solver(model),Solver(model,"gate")
        j,g = joint.value(),gate.value()
        gates.append({"threshold":float(F(k,100)),"joint":float(j),"gate":float(g),
                      "gap":float(g-j),"gate_exact":str(g),
                      "action":gate.action(model.horizon,model.prior,model.initial_state)})
    save_csv("gate_threshold_sweep.csv",gates)

    deadline = []
    for L in (2,4,8,16,32,64,128,256):
        post = shrinking_post_instance(L)
        bounds = old_label_cost_lower_bound(x.pre,post,x.labels,x.costs,L*L,math.exp(-L))
        for i,bound in enumerate(bounds):
            deadline.append({"log_inverse_delta":L,"horizon":L*L,"environment":i+1,
                             "lower_bound":bound,"normalized_lower_bound":bound/L,
                             "separated_coefficient":coefficients[i]["separated_coefficient"],
                             "post_means":str(post)})
    save_csv("shrinking_channel_bound.csv",deadline)

    # Independent policy trees are enumerated once per horizon and reused across priors.
    enumerations = []
    priors = [(F("3/5"),F("1/4"),F("3/20")),
              (F("1/3"),)*3,(F("1/2"),F("1/2"),F(0))]
    for h in range(4):
        model = uploaded_bayes_fixture(horizon=h)
        trees = enumerate_trees(model)
        solver = Solver(model)
        for prior in priors:
            brute = min(dot(prior,tree.costs) for tree in trees)
            dp = solver.value(h,prior,model.initial_state)
            assert brute == dp
            enumerations.append({"horizon":h,"prior":str(tuple(map(str,prior))),
                                 "trees":len(trees),"difference_exact":str(brute-dp),
                                 "value_exact":str(dp)})
        print(f"Three-environment independent enumeration h={h}: {len(trees)} trees",flush=True)
    save_csv("original_enumeration_checks.csv",enumerations)

    base = binary_model(horizon=2,prior="1/2",repair=False,verify_error="0")
    duration_model = expand_durations(base,{("old","VERIFY"):2})
    (ROOT/"configs/B1_duration_example.json").write_text(json.dumps(duration_model.to_dict(),indent=2))

    # Original source included these output CSVs, not the original Python repository.
    reference_dir = ROOT/"provenance/reference_tables"
    differences = []
    with (reference_dir/"bayes_sensitivity.csv").open() as f:
        for old in csv.DictReader(f):
            key = "verify_accuracy" if old["parameter"]=="verification_accuracy" else old["parameter"]
            match = next(r for r in sensitivity if r["parameter"]==key and
                         abs(r["setting"]-float(old["setting"]))<1e-12)
            for oldname,newname in (("jointValue","joint"),("gateValue","gate"),("noRepairValue","no_repair")):
                residue = abs(float(old[oldname])-match[newname])
                assert residue < 1e-10
                differences.append(residue)
    with (reference_dir/"finite_protocols.csv").open() as f:
        for old in csv.DictReader(f):
            name = "joint" if old["policy"]=="joint_decoder" else "separated"
            match = next(r for r in finite if r["policy"]==name and
                         r["environment"]==int(old["environment"])+1)
            for oldname,newname in (("expectedCost","cost"),("meanPreSamples","pre_samples"),
                                    ("terminalError","error")):
                residue = abs(float(old[oldname])-match[newname])
                assert residue < 1e-9
                differences.append(residue)
    summary = {
        "certification_error_below_delta":True,
        "maximum_original_output_difference":max(differences),
        "timeout_note":"Unresolved mass is reported separately from normalization residual",
        "original_enumeration_cases":len(enumerations),
        "largest_original_policy_count":max(r["trees"] for r in enumerations),
        "gate_half_value":next(r["gate"] for r in gates if r["threshold"]==.5),
        "gate_095_value":next(r["gate"] for r in gates if r["threshold"]==.95),
        "finite_E1_cost_ratio":finite[3]["cost"]/finite[0]["cost"],
        "elapsed_seconds":time.perf_counter()-started,
        "certification_arithmetic":"floating point with stable binomial tails",
        "bayes_arithmetic":"exact fractions.Fraction",
        "original_python_repository_available":False,
    }
    (ROOT/"results/integration_report.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    run()
