"""Replay observable synthetic events through the same finite-model solver.

No hidden environment is an input to this adapter. It is not a fitted deployment
integration. Models must remain unchanged for a replay session.
"""
from .belief import successor
from .solver import Solver


class ReplaySession:
    def __init__(self, model, restriction="joint"):
        self.model = model
        self.model_hash = model.digest()
        self.solver = Solver(model, restriction)
        self.remaining = model.horizon
        self.belief = model.prior
        self.state = model.initial_state
        self.history = []

    def _check_model(self):
        if self.model.digest() != self.model_hash:
            raise ValueError("The model changed during replay; start a new session")

    def recommend(self):
        self._check_model()
        decision = self.solver.action(self.remaining, self.belief, self.state)
        return {
            "remaining": self.remaining, "state": self.state,
            "belief": list(map(str, self.belief)),
            "action" if self.remaining else "declaration": decision,
            "expected_remaining_loss": str(
                self.solver.value(self.remaining, self.belief, self.state)),
            "model_hash": self.model_hash,
        }

    def observe(self, observation, next_state):
        """Apply the event following the currently recommended action."""
        self._check_model()
        if self.remaining == 0:
            raise ValueError("The terminal declaration has already been reached")
        recommendation = self.recommend()
        action = recommendation["action"]
        events = [o for o in self.model.kernels[self.state, action]
                  if (o.observation, o.next_state) == (observation, next_state)]
        if len(events) != 1:
            raise ValueError("Event is not in this action's observable kernel")
        probability, posterior = successor(self.belief, events[0])
        self.history.append({
            "before": recommendation,
            "event": {"observation": observation, "next_state": next_state},
            "predictive_probability": str(probability),
        })
        self.remaining -= 1
        self.belief, self.state = posterior, next_state
        return self.recommend()

    def export(self):
        return {
            "model_hash": self.model_hash, "restriction": self.solver.restriction,
            "history": self.history, "current": self.recommend(),
            "data_kind": "observable synthetic finite-model replay",
        }
