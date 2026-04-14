"""
Cases 3 & 4: End-to-end risk map visualization with mock model.

Case 3 — Head-on collision (1D corridor):
  11×11 open grid. Agent 0 (high priority) at (5,1) → goal (5,9).
  Agent 1 (low priority) at (5,9) → goal (5,1). Moving toward each other.

  What this tests:
    - Does agent 1's risk map show high risk on cells between the two agents?
    - Does _propagate_risk spread risk ahead of agent 0's path toward (5,9)?
    - Does the cost function make agent 1 avoid moving toward agent 0?

Case 4 — Perpendicular intersection:
  11×11 open grid. Agent 0 (high priority) at (5,1) → goal (5,9), moving right.
  Agent 1 (low priority) at (1,5) → goal (9,5), moving down. They'd cross at (5,5).

  What this tests:
    - Does the risk map show high risk at/near the intersection (5,5)?
    - Does _propagate_risk spread risk along agent 0's path through the center?
    - Does agent 1 wait or reroute to avoid the crossing?

No real model loaded — _forward_batch returns greedy-biased probabilities
derived from the tokenizer's greedy action string.
Uses the C++ cost2go module for realistic cost-to-go grids.
"""

import sys
import types
from collections import deque
from pathlib import Path

import cppimport.import_hook
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt.inference import MAPFGPTInferenceConfig
from gpt.safe_action_wrapper import (
    ACTION_TO_DELTA,
    DecentralizedWrapper,
    _apply_pos,
    _make_encoder,
)


# ---------------------------------------------------------------------------
# Mock wrapper: skips model loading, returns greedy-biased probs
# ---------------------------------------------------------------------------

class MockWrapper(DecentralizedWrapper):
    """DecentralizedWrapper that skips model loading and mocks forward pass."""

    def __init__(self, cfg, **kwargs):
        # Set all attributes manually — skip super().__init__ which loads model
        self.cfg = cfg
        self.priority_scheme = kwargs.get("priority_scheme", "index")
        self.sim_num_agents = kwargs.get("sim_num_agents", 1)
        self.horizon = kwargs.get("horizon", 3)
        self.epsilon = kwargs.get("epsilon", 0.1)
        self.alpha = kwargs.get("alpha", 0.5)
        self.lambda_1 = kwargs.get("lambda_1", 1.0)
        self.lambda_2 = kwargs.get("lambda_2", 1.0)
        self.sequential_simulation = kwargs.get("sequential_simulation", False)
        self.conflict_radius = kwargs.get("conflict_radius", cfg.agents_radius)

        self.net = None
        self.encoder = _make_encoder(cfg, cfg.num_agents)
        self.sim_encoder = _make_encoder(cfg, self.sim_num_agents)

        self.num_agents = None
        self.priorities = None
        self.cost2go_data = None
        self.action_histories = None
        self.position_histories = None

    def _forward_batch(self, inputs, encoder):
        """Return greedy-biased probs based on the tokenizer's next_action string."""
        if not inputs:
            return torch.empty(0, 5)
        results = []
        for inp in inputs:
            agents = inp.get("agents", [])
            if agents:
                greedy_str = agents[0].get("next_action", "0000")
                probs = _probs_from_greedy(greedy_str)
            else:
                probs = torch.ones(5) / 5
            results.append(probs)
        return torch.stack(results)


def _probs_from_greedy(greedy_str: str) -> torch.Tensor:
    """
    Convert 4-bit greedy string to probability distribution.
    greedy_str[0]=up, [1]=down, [2]=left, [3]=right. '1' = greedy direction.
    """
    # actions: 0=wait, 1=up, 2=down, 3=left, 4=right
    probs = torch.ones(5) * 0.05
    mapping = {0: 1, 1: 2, 2: 3, 3: 4}  # bit index -> action index
    for bit_idx in range(4):
        if bit_idx < len(greedy_str) and greedy_str[bit_idx] == "1":
            probs[mapping[bit_idx]] = 0.7
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# Observation factory
# ---------------------------------------------------------------------------

def make_observations(grid: np.ndarray, agents: list) -> list:
    """
    Build fake observations list.
    agents: list of (pos, target) where pos/target are (x, y) tuples.
    """
    obs = []
    for pos, target in agents:
        obs.append({
            "global_xy": list(pos),
            "global_target_xy": list(target),
            "global_obstacles": grid.copy(),
        })
    return obs


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def visualize_case(
    title: str,
    grid: np.ndarray,
    agents: list,
    risk_map: dict,
    ego_probs: torch.Tensor,
    action_costs: dict,
    filename: str,
):
    """
    Render a 2-panel figure:
      Left: grid heatmap of risk values + agent positions/goals.
      Right: bar chart of action costs for the ego agent.
    """
    rows, cols = grid.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: risk map ---
    risk_grid = np.zeros((rows, cols))
    for (x, y), v in risk_map.items():
        if 0 <= x < rows and 0 <= y < cols:
            risk_grid[x][y] = v

    im = ax1.imshow(risk_grid, cmap="OrRd", vmin=0, vmax=1.0, origin="upper")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Risk")

    # Draw grid lines
    for x in range(rows + 1):
        ax1.axhline(x - 0.5, color="gray", linewidth=0.3)
    for y in range(cols + 1):
        ax1.axvline(y - 0.5, color="gray", linewidth=0.3)

    # Agent positions and goals
    colors = ["blue", "red"]
    labels = ["Agent 0 (high pri)", "Agent 1 (low pri, ego)"]
    for i, ((pos, target), color, label) in enumerate(zip(agents, colors, labels)):
        ax1.plot(pos[1], pos[0], "o", color=color, markersize=14, label=f"{label} pos")
        ax1.plot(target[1], target[0], "*", color=color, markersize=16, label=f"{label} goal")
        # Arrow from pos toward goal
        dy = target[1] - pos[1]
        dx = target[0] - pos[0]
        length = max(abs(dx), abs(dy), 1)
        ax1.annotate(
            "", xy=(pos[1] + dy / length * 0.4, pos[0] + dx / length * 0.4),
            xytext=(pos[1], pos[0]),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )

    # Annotate risk values
    for (x, y), v in risk_map.items():
        if v > 0.01 and 0 <= x < rows and 0 <= y < cols:
            ax1.text(y, x, f"{v:.2f}", ha="center", va="center", fontsize=7, color="black")

    ax1.set_title(f"Risk Map (Agent 1's view)\n{title}")
    ax1.set_xlabel("col (y)")
    ax1.set_ylabel("row (x)")
    ax1.legend(loc="upper right", fontsize=7)

    # --- Right panel: action costs ---
    action_names = ["wait", "up", "down", "left", "right"]
    costs = [action_costs.get(a, float("inf")) for a in range(5)]
    policy_probs = [ego_probs[a].item() for a in range(5)]

    x_pos = np.arange(5)
    bar_width = 0.35

    bars1 = ax2.bar(x_pos - bar_width / 2, costs, bar_width, label="Cost c(a)", color="salmon")
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x_pos + bar_width / 2, policy_probs, bar_width, label="Policy π(a)", color="steelblue", alpha=0.7)

    best_action = min(action_costs, key=action_costs.get)
    ax2.bar(best_action - bar_width / 2, costs[best_action], bar_width, color="darkred", label=f"Selected: {action_names[best_action]}")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(action_names)
    ax2.set_ylabel("Cost c(a) = -λ₁·log π(a) + λ₂·R[dest]", color="salmon")
    ax2_twin.set_ylabel("Policy probability π(a)", color="steelblue")
    ax2.set_title("Action Selection (Agent 1)")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = Path(__file__).parent / filename
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Run a case end-to-end
# ---------------------------------------------------------------------------

def run_case(title, grid, agents, horizon, epsilon, alpha, lambda_1, lambda_2, filename):
    """
    Run the full risk map pipeline for a 2-agent scenario and visualize.
    Agent 0 = high priority, Agent 1 = low priority (ego).
    """
    cfg = MAPFGPTInferenceConfig(
        path_to_weights="weights/model-2M.pt",  # not actually loaded
        device="cpu",
    )
    wrapper = MockWrapper(
        cfg,
        priority_scheme="index",
        sim_num_agents=1,
        horizon=horizon,
        epsilon=epsilon,
        alpha=alpha,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
    )

    observations = make_observations(grid, agents)
    wrapper._initialize(observations)

    # 1. Ego action probs
    all_probs = wrapper.get_action_probs(observations)
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    for i, ((pos, target), _) in enumerate(zip(agents, ["high-pri", "low-pri"])):
        print(f"  Agent {i}: pos={pos} goal={target} probs={all_probs[i].tolist()}")

    # 2. Simulate hp trajectory trees
    trajectories = wrapper._simulate_hp_trajectory_trees(observations, horizon)
    print(f"\n  Trajectory trees computed for agents: {list(trajectories.keys())}")
    for n, occ_list in trajectories.items():
        for t, occ in enumerate(occ_list):
            total = sum(occ.values())
            top = sorted(occ.items(), key=lambda x: -x[1])[:3]
            print(f"    Agent {n}, step t={t+1}: mass={total:.3f}, top cells={top}")

    # 3. Build risk map for ego (agent 1)
    ego_idx = 1
    risk_map = wrapper._build_cumulative_risk_map(ego_idx, observations, trajectories)
    risk_t1 = risk_map.get(1, {})  # t=1: where hp neighbors will be in one step
    total_cells = sum(len(v) for v in risk_map.values())
    print(f"\n  Risk map for agent {ego_idx}: {len(risk_map)} timesteps, {total_cells} total cells")
    for t in sorted(risk_map):
        top = sorted(risk_map[t].items(), key=lambda x: -x[1])[:4]
        print(f"    t={t}: {top}")

    # 4. Action costs for ego (use t=1 risk — where ego arrives in one step)
    ego_pos = tuple(observations[ego_idx]["global_xy"])
    ego_probs = all_probs[ego_idx]
    action_costs = {}
    for a in range(5):
        p = ego_probs[a].item()
        if p < 1e-8:
            action_costs[a] = float("inf")
            continue
        next_pos = _apply_pos(ego_pos, a)
        log_prob = -torch.log(ego_probs[a]).item()
        risk_at_dest = risk_t1.get(next_pos, 0.0)
        cost = lambda_1 * log_prob + lambda_2 * risk_at_dest
        action_costs[a] = cost

    best = min(action_costs, key=action_costs.get)
    action_names = ["wait", "up", "down", "left", "right"]
    print(f"\n  Action costs (using t=1 risk):")
    for a in range(5):
        dest = _apply_pos(ego_pos, a)
        risk_val = risk_t1.get(dest, 0.0)
        marker = " <-- BEST" if a == best else ""
        print(f"    {action_names[a]:5s} -> {dest}  risk={risk_val:.4f}  cost={action_costs[a]:.4f}{marker}")

    visualize_case(title, grid, agents, risk_t1, ego_probs, action_costs, filename)
    return risk_map, action_costs


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case3_head_on():
    """
    Head-on collision in a corridor (same row).
    Agent 0: (10,7) → (10,13), Agent 1: (10,13) → (10,7).
    21×21 grid so all positions/targets are within cost2go_radius=5 from border.
    """
    grid = np.zeros((21, 21), dtype=int)
    agents = [
        ((10, 7), (10, 13)),  # Agent 0: high priority, moving right
        ((10, 13), (10, 7)),  # Agent 1: low priority, moving left
    ]
    return run_case(
        title="Case 3: Head-On Collision",
        grid=grid,
        agents=agents,
        horizon=3,
        epsilon=0.1,
        alpha=0.5,
        lambda_1=1.0,
        lambda_2=1.0,
        filename="case3_head_on.png",
    )


def case4_perpendicular():
    """
    Perpendicular intersection.
    Agent 0: (10,7) → (10,13), moving right through center.
    Agent 1: (7,10) → (13,10), moving down through center.
    They cross near (10,10).
    """
    grid = np.zeros((21, 21), dtype=int)
    agents = [
        ((10, 7), (10, 13)),  # Agent 0: high priority, moving right
        ((7, 10), (13, 10)),  # Agent 1: low priority, moving down
    ]
    return run_case(
        title="Case 4: Perpendicular Intersection",
        grid=grid,
        agents=agents,
        horizon=3,
        epsilon=0.1,
        alpha=0.5,
        lambda_1=1.0,
        lambda_2=1.0,
        filename="case4_perpendicular.png",
    )


if __name__ == "__main__":
    case3_head_on()
    case4_perpendicular()
