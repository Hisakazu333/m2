"""A sufficient, checked environment aggregation; no minimality claim."""
from copy import deepcopy
from .model import Model, Outcome


def aggregate_environments(model, blocks):
    flat = [i for block in blocks for i in block]
    if sorted(flat) != list(range(len(model.environments))) or any(not b for b in blocks):
        raise ValueError("Blocks must partition every environment exactly once")
    def reduce_equal(values):
        result = []
        for block in blocks:
            value = values[block[0]]
            if any(values[i] != value for i in block):
                raise ValueError("The proposed block is not intervention sufficient")
            result.append(value)
        return tuple(result)
    states = deepcopy(model.states)
    for s, data in states.items():
        data["labels"] = list(reduce_equal(data["labels"]))
    kernels = {key: tuple(Outcome(o.observation, o.next_state,
                                  reduce_equal(o.probabilities),
                                  reduce_equal(o.task), reduce_equal(o.direct))
                          for o in outcomes)
               for key, outcomes in model.kernels.items()}
    terminal = {s: {d: reduce_equal(costs) for d, costs in decisions.items()}
                for s, decisions in model.terminal.items()}
    prior = tuple(sum(model.prior[i] for i in block) for block in blocks)
    metadata = deepcopy(model.metadata)
    metadata["environment_aggregation"] = [list(block) for block in blocks]
    return Model(tuple(f"class_{j}" for j in range(len(blocks))), states, model.actions.copy(),
                 kernels, terminal, prior, model.initial_state, model.horizon, metadata).validate()
