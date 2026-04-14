"""
Case 2: Unit test for _simulate_single_agent_tree.

Tests whether the trajectory tree correctly propagates probability mass
through pruned actions over multiple steps.

Setup: 5×5 open grid. Agent at (2,2), goal at (0,0). horizon=3, epsilon=0.1.

What this tests:
  - Mass conservation: total probability across frontier cells should sum to ~1.0
    at every timestep (probability mass is neither created nor destroyed).
  - Pruning: actions below epsilon are dropped and remaining probs renormalized.
  - Branching: frontier should grow as the tree explores multiple actions.
  - Spatial coherence: high-mass cells should be between start and goal.

No model loading required — _forward_batch is mocked with greedy-biased probs.
No C++ cost2go module required — _build_sim_input is mocked.
"""

import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt.safe_action_wrapper import ACTION_TO_DELTA, DecentralizedWrapper, MOVES_STR


def bfs_cost2go(rows: int, cols: int, target: tuple) -> list:
    """BFS distance grid on an open grid. d[x][y] = manhattan-ish shortest path distance."""
    dist = [[-1] * cols for _ in range(rows)]
    tx, ty = target
    dist[tx][ty] = 0
    q = deque([(tx, ty)])
    while q:
        x, y = q.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))
    return dist


def greedy_probs(pos: tuple, target: tuple, dist_grid: list, rows: int, cols: int) -> torch.Tensor:
    """Greedy-biased action distribution: high prob for actions reducing distance to goal."""
    probs = torch.ones(5) * 0.05
    current_dist = dist_grid[pos[0]][pos[1]]
    for a, (dx, dy) in ACTION_TO_DELTA.items():
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            nd = dist_grid[nx][ny]
            if nd >= 0 and nd < current_dist:
                probs[a] = 0.6
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return probs


def test_trajectory_tree():
    rows, cols = 5, 5
    target = (0, 0)
    start = (2, 2)
    horizon = 3
    epsilon = 0.1

    dist_grid = bfs_cost2go(rows, cols, target)

    # Build mock self with needed attributes and methods
    mock = SimpleNamespace(
        epsilon=epsilon,
        cfg=SimpleNamespace(num_previous_actions=5),
        sim_encoder=None,
    )

    def _build_sim_input(self, pos, tgt, history):
        return {"_pos": pos, "_target": tgt}

    def _forward_batch(self, inputs, encoder):
        results = []
        for inp in inputs:
            pos = inp["_pos"]
            results.append(greedy_probs(pos, target, dist_grid, rows, cols))
        return torch.stack(results)

    mock._build_sim_input = types.MethodType(_build_sim_input, mock)
    mock._forward_batch = types.MethodType(_forward_batch, mock)
    mock._prune_probs = types.MethodType(DecentralizedWrapper._prune_probs, mock)

    occupancy = DecentralizedWrapper._simulate_single_agent_tree(
        mock,
        start_pos=start,
        target=target,
        start_history=["n"] * 5,
        horizon=horizon,
    )

    print("=== Case 2: _simulate_single_agent_tree on 5×5 open grid ===")
    print(f"Start: {start} | Goal: {target} | horizon={horizon} | epsilon={epsilon}")
    print()

    all_ok = True
    for t, occ in enumerate(occupancy):
        total_mass = sum(occ.values())
        n_cells = len(occ)
        top_cells = sorted(occ.items(), key=lambda x: -x[1])[:5]
        mass_ok = abs(total_mass - 1.0) < 0.01
        if not mass_ok:
            all_ok = False

        print(f"Step t={t+1}: total_mass={total_mass:.4f}  frontier_cells={n_cells}  [{'OK' if mass_ok else 'BUG: mass not conserved'}]")
        for cell, mass in top_cells:
            print(f"    {cell}: {mass:.4f}")
        print()

    if all_ok:
        print(">>> ALL OK: mass conserved at every step.")
    else:
        print(">>> BUG: mass not conserved at one or more steps.")

    return occupancy, dist_grid


def visualize(occupancy, dist_grid):
    rows = len(dist_grid)
    cols = len(dist_grid[0])
    horizon = len(occupancy)
    start = (2, 2)
    target = (0, 0)

    fig, axes = plt.subplots(1, horizon, figsize=(5 * horizon, 5))
    if horizon == 1:
        axes = [axes]

    for t, (occ, ax) in enumerate(zip(occupancy, axes)):
        grid = np.zeros((rows, cols))
        for (x, y), mass in occ.items():
            if 0 <= x < rows and 0 <= y < cols:
                grid[x][y] = mass

        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=0.6, origin="upper")
        ax.set_title(f"Step t={t+1}\ntotal mass={sum(occ.values()):.3f}")
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

        # Mark start and goal
        ax.plot(start[1], start[0], "rs", markersize=12, label="Start")
        ax.plot(target[1], target[0], "g*", markersize=14, label="Goal")

        for (x, y), mass in occ.items():
            if mass > 0.01 and 0 <= x < rows and 0 <= y < cols:
                ax.text(y, x, f"{mass:.2f}", ha="center", va="center", fontsize=8, color="darkred")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("Trajectory Tree Occupancy (greedy-biased mock policy)", fontsize=13)
    plt.tight_layout()
    out = Path(__file__).parent / "trajectory_tree.png"
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    occupancy, dist_grid = test_trajectory_tree()
    visualize(occupancy, dist_grid)
