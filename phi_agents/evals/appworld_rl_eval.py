#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from collections import defaultdict
from collections.abc import Sequence
from typing import cast

import numpy as np

from phi_agents.evals.appworld_rollout_data import AppWorldTrainingRollout
from phi_agents.rl.eval import RolloutSummaries
from phi_agents.rl.type_defs import Scenario, TrainingRollout


class AppworldEvalSummaries(RolloutSummaries):
    def summary(
        self, scenarios: list[Scenario], rollouts: Sequence[TrainingRollout]
    ) -> dict[str, float]:
        rollouts = cast(Sequence[AppWorldTrainingRollout], rollouts)
        # sc = cast(list[AppWorldScenario], scenarios)
        rs = rollouts

        # Assume ret is binary (0 or 1)
        rets_by_difficulty: dict[int, list[float]] = defaultdict(list)
        for r in rs:
            rets_by_difficulty[r.appworld_rollout_data.eval_result.difficulty].append(r.ret)
        tgc_by_difficulty: dict[str, float] = {
            f"task_goal_completion__dfcty{dfcty}": sum(rets) / len(rets)
            for dfcty, rets in rets_by_difficulty.items()
        }

        stats = {
            "task_goal_completion": float(np.mean([r.ret for r in rs])),
            **tgc_by_difficulty,
        }

        return {f"appworld_eval/{k}": v for k, v in stats.items()}
