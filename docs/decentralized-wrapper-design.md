# Decentralized Safe Action Wrapper — Design Document

## Motivation

Safety-shield methods like CS-PIBT require each agent to know the exact next step
of other agents. In real-world decentralized multi-agent systems, this requires
communication — which introduces latency and noise.

**Key insight**: if all robots run the same policy network with the same weights,
each agent can **forward-simulate** what its neighbors will do, without any
communication. Combined with a fixed priority ordering (known to all agents at
episode start), this gives each agent enough information to avoid collisions
autonomously.

## Architecture

### Files

| File | Purpose |
|------|---------|
| `gpt/safe_action_wrapper.py` | `DecentralizedWrapper` class |
| `example_safe.py` | Runnable example (drop-in replacement for `example.py`) |

### Class: `DecentralizedWrapper`

Drop-in replacement for `MAPFGPTInference`. Same `act(observations)` /
`reset_states()` interface, compatible with `run_episode()`.

```
DecentralizedWrapper
├── __init__(cfg, priority_scheme, sim_num_agents, horizon, epsilon, alpha, lambda_1, lambda_2, sequential_simulation)
├── act(observations) → List[int]              # main entry point
├── act_with_info(observations) → dict         # with diagnostics
├── get_action_probs(observations) → (N,5)     # raw policy probs
├── simulate_neighbor(neighbor, ego, obs)      # single neighbor (1 step)
├── simulate_visible_neighbors(ego, obs)       # batch neighbors (1 step)
├── get_safe_action(obs, do_sample)            # core safety logic
│   ├── _get_safe_action_single_step()         # horizon=1: hard mask
│   └── _get_safe_action_multistep()           # horizon>1: risk map
├── _simulate_hp_trajectory_trees(obs, h)      # trajectory tree simulation
│   └── _simulate_single_agent_tree(...)       # per-agent tree propagation
├── _build_cumulative_risk_map(ego, obs, traj) # per-ego risk map (Eq. 3)
├── _propagate_risk(risk_map, target)          # implicit risk propagation (Eq. 4)
├── _prune_probs(probs)                        # action pruning (Eq. 1)
├── _build_sim_input(pos, target, history)     # input at simulated position
└── _build_input(ego, obs, max_agents, ctx)    # input from real observations
```

## How It Works

### Assumptions

1. All agents share the **same policy network** (same weights).
2. All agents know visible neighbors' **goals and IDs**.
3. A **fixed priority** ordering is assigned at episode start and known to all.
4. Only agents within **observation radius** are relevant for conflict checks —
   distant agents are ignored.

### Two Modes

The wrapper supports two safety modes controlled by the `horizon` parameter:

| Mode | `horizon` | Approach | When to use |
|------|-----------|----------|-------------|
| Single-step | `1` | Hard mask: zero out conflicting actions | Fast, few agents |
| Multi-step | `>1` | Risk map from trajectory tree simulation | Many agents, dense scenarios |

---

### Mode 1: Single-Step Hard Mask (`horizon=1`)

All agents commit **simultaneously**. Priority is a **tie-breaker**, not a
processing order.

```
For each agent i (all in parallel):
  1. Run shared policy with full context → action probs P_i (shape: 5)

  2. Identify higher-priority visible neighbors H_i

  3. For each neighbor j ∈ H_i:
     - Build j's observation using reduced context window
       (only agents visible to BOTH i and j)
     - Run shared policy → probs P_j
     - Predict j's action = argmax(P_j)
     - Compute j's predicted next position

  4. For each candidate action a ∈ {wait, up, down, left, right}:
     - Compute ego's next position if taking action a
     - Check for vertex conflict (same cell as any predicted neighbor pos)
     - Check for edge conflict (swapping cells with any neighbor)
     - If conflict: mask out action a (set prob to 0)

  5. Re-normalize remaining probs, sample action
     (fall back to wait if all actions masked)
```

---

### Mode 2: Risk Map via Trajectory Tree (`horizon>1`)

Single-step is reactive — it only catches imminent collisions. With many agents,
avoiding a collision at step t often pushes it to step t+1. The multi-step mode
builds a **cumulative risk map** from trajectory tree simulation and selects
actions that balance policy preference against collision risk.

#### Step 1: Ego Action Probabilities

Run the shared policy on all N agents with full context (13-agent window) in one
batched forward pass → `P_policy` of shape `(N, 5)`.

#### Step 2: Simulate Higher-Priority Neighbors' Trajectory Trees (Eq. 1-2)

For each unique higher-priority neighbor, build a **trajectory tree** for `h`
steps. Unlike the previous argmax-advance approach, the tree propagates
probability mass through **all actions above threshold ε**:

```
For neighbor p_ℓ at position (i₀, j₀):
  frontier = {(i₀, j₀): probability 1.0}

  For each step t = 0..h-1:
    1. Run shared policy at each frontier position
    2. Prune: zero actions with probability < ε, renormalize (Eq. 1):
       π̃(a | o) = π(a | o) / Σ_{a': π(a'|o) ≥ ε} π(a' | o)   if π(a | o) ≥ ε
                 = 0                                              otherwise
    3. Propagate mass: for each frontier cell and each surviving action,
       compute next position and accumulate probability mass (Eq. 2):
       r_{ijt} = Σ_{(i',j') ∈ N(i,j)} r_{i'j',t-1} · π̃(a_{(i',j')→(i,j)} | o)
    4. Record step occupancy: occupancy[t][(x,y)] = total mass at (x,y)
    5. New frontier = all cells with nonzero mass
```

This produces a **full probability distribution over cells** at each timestep,
not just a single predicted position.

Example for a neighbor at (3,5) with policy [0.05, 0.1, 0.7, 0.1, 0.05] and ε=0.1:

```
After pruning (ε=0.1): [0, 0.111, 0.778, 0.111, 0]  (wait and right pruned)
Step t occupancy:
  (2,5) → 0.111  (up)
  (4,5) → 0.778  (down)
  (3,4) → 0.111  (left)

Next step: frontier has 3 positions, each gets its own policy evaluation
```

**Action history**: Each frontier position keeps the history from its
highest-mass predecessor path (simplified tracking).

**Simulation modes** (`sequential_simulation` parameter):

- **Independent** (`False`, default): All hp neighbors simulated independently.
  All frontier positions across all agents batched into one forward pass per step.
- **Sequential** (`True`): Simulate in priority order. Each agent p_k's tree
  accounts for occupancy from already-simulated p_1..p_{k-1} — actions landing
  on high-occupancy cells are penalized before tree propagation.

#### Step 3: Build Cumulative Risk Map (Eq. 3)

For each ego agent, aggregate per-neighbor occupancy into a cumulative risk map:

1. For each hp neighbor, merge occupancy across all timesteps (max per cell).
2. Propagate risk along the neighbor's distance-to-goal path (Eq. 4, see below).
3. Aggregate across neighbors using **element-wise max** (not sum):

```
R_{ijt} = max_{p_ℓ ∈ P} r_{ijt}^{(p_ℓ)}    ∀ i, j, t
```

The max operation reflects that collision risk is dominated by the single most
likely conflicting agent, not a sum of independent probabilities.

#### Step 4: Implicit Risk Propagation (Eq. 4)

The raw risk map captures direct occupancy but doesn't account for the likelihood
that a higher-priority agent will pass through a cell **on its way** to its goal.
Risk is propagated along each neighbor's distance-to-goal map with decay factor α:

```
For each cell (i,j) with risk R_{ijt}:
  For each neighboring cell (x,y) closer to the goal (d_{xy} < d_{ij}):
    R_{xyt} ← max(R_{xyt}, R_{ijt} · α^{d_{ij} - d_{xy}})
```

This ensures cells along the agent's likely future path carry risk even if
they weren't directly in the trajectory tree's horizon.

#### Step 5: Risk-Aware Action Selection (Eq. 5-6)

For each candidate action, compute a cost that balances policy preference against
collision risk:

```
c(a) = -λ₁ · log π*(a | oᵢ) + λ₂ · R_{xy,τ+1}
```

where `-log π*(a | oᵢ)` is the negative log-probability (lower is more preferred)
and `R_{xy,τ+1}` is the risk at the destination cell. The agent selects:

```
a* = argmin_{a ∈ A(i,j)} c(a)
```

- `lambda_1` controls how much the agent follows the learned policy
- `lambda_2` controls how much the agent avoids risky cells

#### Batching Strategy

**Independent mode** (`sequential_simulation=False`):

| Pass | Batch size | Count |
|------|-----------|-------|
| Ego action probs (full context) | N | 1 |
| Neighbor trajectory tree (all frontiers) | total frontier cells | h |
| **Total forward passes** | | **h + 1** |

**Sequential mode** (`sequential_simulation=True`):

| Pass | Batch size | Count |
|------|-----------|-------|
| Ego action probs (full context) | N | 1 |
| Neighbor trajectory tree (per agent) | per-agent frontier cells | h × K |
| **Total forward passes** | | **h × K + 1** |

where K = number of unique hp neighbors.

No ego rollout forward passes are needed — the risk map with implicit
propagation replaces them entirely.

---

### Why It Works Without Communication

Agent i predicts agent j's action by running the **same policy** on an
approximation of j's observation. Since j is also running the same policy,
the prediction matches j's actual decision (assuming deterministic argmax).

Priority ensures consistency: agent j (higher priority) doesn't need to
worry about agent i. Agent i knows this, so i's prediction of j doesn't
depend on what i itself does — no circular dependency.

### Dual Encoder Design

Two `Encoder` instances with different `num_agents` settings:

| Encoder | `num_agents` | Used for |
|---------|-------------|----------|
| `self.encoder` | 13 (default) | Ego agent's own action — full context |
| `self.sim_encoder` | `sim_num_agents` (default 1) | Simulating neighbors — reduced window |

**Token count comparison** (with default parameters):

| Component | Full (13 agents) | Reduced (1 agent) |
|-----------|------------------|--------------------|
| Cost2go grid | 121 | 121 |
| Agent slots | 130 (13 × 10) | 10 (1 × 10) |
| **Total meaningful** | **251** | **131** |

The reduced window makes simulation cheaper (fewer tokens to process)
while retaining the cost2go spatial context. The `sim_num_agents` parameter
controls this tradeoff.

## Usage

### Basic (drop-in, multi-step by default)

```bash
python example_safe.py \
  --map_name validation-mazes-seed-000 \
  --model 2M \
  --num_agents 32 \
  --device mps
```

### Single-step fallback

```bash
python example_safe.py \
  --model 2M \
  --num_agents 32 \
  --device mps \
  --horizon 1
```

### Tuning risk map parameters

```bash
python example_safe.py \
  --model 2M \
  --num_agents 64 \
  --device mps \
  --horizon 5 \
  --epsilon 0.1 \
  --alpha 0.5 \
  --lambda_1 1.0 \
  --lambda_2 2.0 \
  --priority_scheme random
```

### Sequential simulation mode

```bash
python example_safe.py \
  --model 2M \
  --num_agents 32 \
  --device mps \
  --horizon 3 \
  --sequential
```

### Programmatic

```python
from gpt.inference import MAPFGPTInferenceConfig
from gpt.safe_action_wrapper import DecentralizedWrapper

cfg = MAPFGPTInferenceConfig(path_to_weights="weights/model-2M.pt", device="mps")
wrapper = DecentralizedWrapper(
    cfg,
    priority_scheme="index",
    sim_num_agents=1,
    horizon=3,
    epsilon=0.1,
    alpha=0.5,
    lambda_1=1.0,
    lambda_2=1.0,
    sequential_simulation=False,
)
wrapper.reset_states()

# In episode loop:
actions = wrapper.act(observations)

# Or with diagnostics:
info = wrapper.act_with_info(observations)
info["actions"]         # List[int] — safe actions
info["probs"]           # Tensor (N, 5) — raw policy probs
info["priorities"]      # List[int] — priority assignments
info["next_positions"]  # Dict[int, tuple] — chosen next positions
```

### Lower-level API

```python
# Raw action probabilities (full context)
probs = wrapper.get_action_probs(observations)  # (N, 5)

# Simulate a specific neighbor (single step)
neighbor_probs = wrapper.simulate_neighbor(
    neighbor_idx=3, ego_idx=0, observations=observations
)  # (5,)

# Simulate all higher-priority visible neighbors (single step)
neighbor_dict = wrapper.simulate_visible_neighbors(
    ego_idx=5, observations=observations, only_higher_priority=True
)  # {neighbor_idx: (5,) tensor}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `priority_scheme` | `"index"` | `"index"`: agent 0 = highest priority. `"random"`: shuffled at episode start. |
| `sim_num_agents` | `1` | Agent slots in reduced window for simulation. 1 = only the agent itself. Higher = richer context but slower. |
| `horizon` | `3` | Steps to simulate forward. 1 = single-step hard mask. >1 = risk map approach. |
| `epsilon` | `0.1` | Pruning threshold for trajectory tree. Actions below this probability are dropped (Eq. 1). |
| `alpha` | `0.5` | Decay factor for implicit risk propagation along distance-to-goal paths (Eq. 4). |
| `lambda_1` | `1.0` | Weight for policy log-probability in cost function. Higher = follow policy more (Eq. 5). |
| `lambda_2` | `1.0` | Weight for risk map in cost function. Higher = more conservative avoidance (Eq. 5). |
| `sequential_simulation` | `False` | If True, simulate hp neighbors in priority order with mutual avoidance. |

## Limitations and Future Work

1. **Approximate neighbor simulation**: When ego simulates neighbor j, it can only
   include agents that ego itself can see. Neighbor j might see additional agents
   outside ego's radius. This is a fundamental decentralized constraint.

2. **Compounding prediction error**: Each simulation step builds on the previous
   step's predicted position. By step h, the predicted trajectory may diverge from
   reality. Probability pruning (ε) helps control tree growth but the frontier
   can still drift. In practice h=3–5 is a sweet spot.

3. **Same-priority conflicts**: Agents only yield to strictly higher-priority
   neighbors. Two agents with adjacent priorities that can't see each other may
   still collide. This is inherent to the decentralized setting.

4. **Simplified action history in tree**: Each frontier position keeps the history
   from its highest-mass predecessor. Different paths to the same cell may have
   different histories; we approximate with the dominant one. The cost2go grid
   dominates the policy's spatial reasoning, so this has marginal impact.

5. **Static other-agent positions during simulation**: When simulating a neighbor
   forward, other agents' positions are not updated (they stay at their t=0
   positions). This is an approximation that could be improved with joint
   simulation, at the cost of complexity.

## Git History

Branch: `feat/safe-action-wrapper`

```
fb1637f docs: Update design doc with multi-step cost field approach
3797d9e feat: Add multi-step probabilistic occupancy cost field
fb638a9 refactor: Make all agents commit simultaneously, priority as tie-breaker only
7d24a7c feat: Rewrite as fully decentralized wrapper with fixed priorities
bbc9d5e feat: Add SafeActionWrapper for conflict-free decentralized action selection
```
