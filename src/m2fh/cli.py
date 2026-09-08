from pathlib import Path
import argparse
import json
from .model import load_model
from .solver import Solver
from .policies import diagnostics


def main():
    parser = argparse.ArgumentParser(description="Exact finite-horizon M2 solver")
    parser.add_argument("config")
    parser.add_argument("--policy", default="joint", choices=["joint", "gate", "never_repair"])
    parser.add_argument("--output", default="results/solution.json")
    args = parser.parse_args()
    model = load_model(args.config)
    solver = Solver(model, args.policy)
    result = {"value": str(solver.value()), "value_float": float(solver.value()),
              "policy": solver.export_policy(), "diagnostics": diagnostics(model, solver.action)}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"value={result['value']} ({result['value_float']:.8g}); saved {target}")


if __name__ == "__main__":
    main()
