"""Produce a deterministic, visible-event example; no hidden truth is sampled."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from m2fh.model import load_model, dot
from m2fh.replay import ReplaySession

model = load_model(ROOT/"configs/F5_failed_repair.json")
session = ReplaySession(model)
while session.remaining:
    action = session.recommend()["action"]
    # Choose the most likely visible event solely to make this example repeatable.
    outcome = max(model.kernels[session.state, action],
                  key=lambda o: dot(session.belief, o.probabilities))
    session.observe(outcome.observation, outcome.next_state)
target = ROOT/"results/replay_example.json"
target.write_text(json.dumps(session.export(), indent=2), encoding="utf-8")
print(f"Replay written to {target}")
