"""Independently reconstruct the uploaded PDF's two numerical fixtures."""
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
import csv
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from m2fh.certification import RepairExperiment, finite_protocol
from m2fh.pdf_fixture import uploaded_bayes_fixture, RepairOnlyGate
from m2fh.solver import Solver
from m2fh.belief import branches

def conditional_errors(model,solver):
    @lru_cache(None)
    def rec(i,h,b,state):
        if not h:
            d=model.terminal_value(b,state)[1]
            return Fraction(int(int(d)!=model.states[state]["labels"][i]))
        action=solver.action(h,b,state)
        return sum((o.probabilities[i]*rec(i,h-1,post,o.next_state)
                    for _,post,o,_ in branches(model,b,state,action)),Fraction(0))
    return [float(rec(i,model.horizon,model.prior,model.initial_state))
            for i in range(len(model.prior))]

def main():
    OUT=ROOT/"results"
    rows=[]
    m=uploaded_bayes_fixture()
    for name,s in (("joint",Solver(m)),("posterior_gate",RepairOnlyGate(m)),
                   ("no_repair",Solver(m,"never_repair"))):
        rows.append({"policy":name,"value":float(s.value()),"exact_value":str(s.value()),
                     "action":s.action(m.horizon,m.prior,m.initial_state),
                     "conditional_errors":conditional_errors(m,s)})
    (OUT/"uploaded_pdf_bayes_audit.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2),flush=True)
    x=RepairExperiment()
    result=finite_protocol(x)+finite_protocol(x,separated=True)
    with (OUT/"uploaded_pdf_certification_audit.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (OUT/"uploaded_pdf_coefficients.json").write_text(json.dumps(x.coefficients(),indent=2))
    print(json.dumps(result,indent=2),flush=True)
    print("Uploaded PDF fixtures independently reconstructed; original source code was not available.")

if __name__=="__main__":
    main()
