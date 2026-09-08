"""One command rebuilds all results, checks, figures and the integrated PDF."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--skip-pdf",action="store_true")
    args=parser.parse_args()
    if not args.skip_pdf and (not shutil.which("latexmk") or not shutil.which("pdflatex")):
        parser.error("Install latexmk and pdflatex, or explicitly pass --skip-pdf")
    logs=ROOT/"results/logs";logs.mkdir(parents=True,exist_ok=True)
    stages=[
        ("tests",[sys.executable,"scripts/run_tests.py"]),
        ("integrated_experiments",[sys.executable,"scripts/run_integrated_experiments.py"]),
        ("binary_experiments",[sys.executable,"scripts/run_experiments.py"]),
        ("binary_figures",[sys.executable,"scripts/make_figures.py"]),
        ("integrated_figures",[sys.executable,"scripts/make_integrated_figures.py"]),
        ("replay",[sys.executable,"examples/replay_example.py"]),
        ("cli",[sys.executable,"-m","m2fh.cli","configs/B0_original_bayes.json",
                "--policy","gate","--output","results/cli_example.json"]),
    ]
    if not args.skip_pdf:
        stages.append(("latex",["latexmk","-cd","-pdf","-interaction=nonstopmode",
                                "-halt-on-error","paper/main.tex"]))
    env=os.environ.copy();env["PYTHONPATH"]=str(ROOT/"src")
    env["MPLBACKEND"]="Agg"
    report={"started_utc":datetime.now(timezone.utc).isoformat(),
            "python":platform.python_version(),"platform":platform.platform(),
            "dependencies":{n:importlib.metadata.version(n) for n in ("numpy","matplotlib")},
            "stages":[],"success":False}
    start=time.perf_counter()
    path=ROOT/"results/validation_report.json"
    for name,command in stages:
        print(f"Running {name} ...",flush=True)
        t=time.perf_counter()
        result=subprocess.run(command,cwd=ROOT,env=env,text=True,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (logs/f"{name}.txt").write_text(result.stdout,encoding="utf-8")
        report["stages"].append({"name":name,"command":command,
                                "returncode":result.returncode,"seconds":time.perf_counter()-t})
        path.write_text(json.dumps(report,indent=2))
        if result.returncode:
            print(result.stdout,file=sys.stderr);raise SystemExit(result.returncode)
        print(f"{name}: passed",flush=True)
    if not args.skip_pdf:
        target=ROOT/"output/pdf/M2_Integrated_Manuscript.pdf";target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"paper/main.pdf",target)
        log=(ROOT/"paper/main.log").read_text()
        if "undefined references" in log or "Overfull" in log:
            raise SystemExit("Inspect LaTeX references/layout warnings before release")
    report["success"]=True;report["elapsed_seconds"]=time.perf_counter()-start
    report["source_sha256"]=json.loads((ROOT/"results/metadata.json").read_text())["source_sha256"]
    path.write_text(json.dumps(report,indent=2))
    print("All requested stages passed; see results/validation_report.json")


if __name__=="__main__":
    main()
