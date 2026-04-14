# DecentralizedWrapper Diagnostic Tests

These tests visualize the internals of `DecentralizedWrapper` to identify bugs.
No real model is loaded — all neural network calls are mocked with greedy-biased heuristics.

## Test Cases

### Case 1: `test_propagate_risk.py` — Risk Propagation Unit Test

**Purpose**: Test whether `_propagate_risk` spreads risk transitively along
the distance-to-goal path, or stops after one hop.

**Setup**: 1×7 corridor. Risk source at cell (0,6), goal at (0,0), α=0.5.

**Expected**: Risk decays as α^d along the path: (0,5)=0.5, (0,4)=0.25, ..., (0,0)=0.016.

**Finding**: BUG — only (0,5) gets risk. The loop iterates over the original
`risk_map` entries and never re-processes newly propagated cells.

---

### Case 2: `test_trajectory_tree.py` — Trajectory Tree Mass Conservation

**Purpose**: Verify that `_simulate_single_agent_tree` conserves probability
mass across timesteps (total mass should remain ~1.0) and that the tree
branches correctly through pruned actions.

**Setup**: 5×5 open grid. Agent at (2,2), goal at (0,0). horizon=3, ε=0.1.

**Expected**: Mass sums to 1.0 at every step. Frontier grows as the tree
explores multiple surviving actions.

**Finding**: OK — mass is conserved. Tree branches correctly toward the goal.

---

### Case 3: `visualize_risk_map.py` → `case3_head_on()` — Head-On Collision

**Purpose**: Test the full risk map pipeline when two agents approach each
other head-on. The low-priority agent (Agent 1) should see high risk ahead
and avoid moving toward the high-priority agent (Agent 0).

**Setup**: 21×21 open grid. Agent 0 at (10,7)→(10,13), Agent 1 at (10,13)→(10,7).
They are 6 cells apart on the same row, moving toward each other.

**Expected**: Agent 1's risk map should show high risk on cells between the
two agents. Agent 1 should wait or detour instead of moving left.

**Finding**: BUG — risk map is completely empty. Agents are 6 cells apart,
which exceeds `agents_radius=5`, so `_get_visible_neighbors` returns nothing.
No trajectory tree is built. Agent 1 moves directly toward Agent 0 with
zero collision awareness.

---

### Case 4: `visualize_risk_map.py` → `case4_perpendicular()` — Perpendicular Intersection

**Purpose**: Test the risk map when two agents cross perpendicularly. Agent 0
moves right through (10,10); Agent 1 moves down through (10,10). The risk map
should flag the intersection area as dangerous.

**Setup**: 21×21 open grid. Agent 0 at (10,7)→(10,13), Agent 1 at (7,10)→(13,10).
They would cross near (10,10). Distance = 3, within visibility.

**Expected**: Risk map should show high risk at (10,8), (10,9), (10,10) from
the trajectory tree, plus propagated risk toward (10,13) along Agent 0's path.

**Finding**: Trajectory tree works (occupancy at (10,8), (10,9), (10,10)).
Risk propagation only adds one cell beyond the tree ((10,11)=0.5) due to the
one-hop bug. Agent 1 still selects "down" (correct in this case since it's
far from the intersection), but the weak propagation would fail if Agent 1
were closer.

## Running

```bash
python tests/test_propagate_risk.py
python tests/test_trajectory_tree.py
python tests/visualize_risk_map.py
```

Output images are saved to `tests/*.png`.
