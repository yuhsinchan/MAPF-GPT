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
├── __init__(cfg, priority_scheme, sim_num_agents, horizon, gamma, safety_lambda)
├── act(observations) → List[int]              # main entry point
├── act_with_info(observations) → dict         # with diagnostics
├── get_action_probs(observations) → (N,5)     # raw policy probs
├── simulate_neighbor(neighbor, ego, obs)      # single neighbor (1 step)
├── simulate_visible_neighbors(ego, obs)       # batch neighbors (1 step)
├── get_safe_action(obs, do_sample)            # core safety logic
│   ├── _get_safe_action_single_step()         # horizon=1: hard mask
│   └── _get_safe_action_multistep()           # horizon>1: cost field
├── _simulate_hp_trajectories(obs, horizon)    # multi-step neighbor rollout
├── _build_ego_occupancy(ego, obs, traj)       # per-ego occupancy field
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
| Multi-step | `>1` | Soft cost field from probabilistic occupancy | Many agents, dense scenarios |

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

### Mode 2: Multi-Step Probabilistic Cost Field (`horizon>1`)

Single-step is reactive — it only catches imminent collisions. With many agents,
avoiding a collision at step t often pushes it to step t+1. The multi-step mode
builds a **probabilistic occupancy field** over future timesteps and penalizes
actions whose trajectories overlap with it.

#### Step 1: Ego Action Probabilities

Run the shared policy on all N agents with full context (13-agent window) in one
batched forward pass → `P_policy` of shape `(N, 5)`.

#### Step 2: Simulate Higher-Priority Neighbors' Trajectories

For each unique higher-priority neighbor, simulate `h` steps forward:

```
At each step t:
  1. Run shared policy → full distribution [p_wait, p_up, p_down, p_left, p_right]
  2. Record occupancy: each reachable cell gets the probability mass
     of the action leading there
  3. Advance position by argmax for the next step's simulation
```

This produces a **soft occupancy field**: cells along the most likely path get
high probability mass, but adjacent cells also get nonzero mass from alternative
actions. This is much richer than a deterministic argmax-only prediction.

Example for a neighbor at (3,5) with policy output [0.05, 0.1, 0.7, 0.1, 0.05]:

```
Step t occupancy:
  (3,5) → 0.05  (wait)
  (2,5) → 0.10  (up)
  (4,5) → 0.70  (down)    ← argmax, used to advance
  (3,4) → 0.10  (left)
  (3,6) → 0.05  (right)
```

The neighbor advances to (4,5) for step t+1, but the cost field remembers that
(2,5), (3,4), and (3,6) each had some probability mass.

#### Step 3: Build Per-Ego Occupancy

For each ego agent, sum occupancy probabilities from all its higher-priority
visible neighbors at each cell and step:

```
occupancy[t][(x,y)] = Σ_j P(neighbor j at (x,y) at step t)
```

#### Step 4: Simulate Ego Candidate Trajectories

For each of the 5 candidate actions, simulate the ego forward for `h` steps:

```
For action a ∈ {0,1,2,3,4}:
  1. Take action a → ego at pos_1
  2. Run policy at pos_1 → advance by argmax → pos_2
  3. Repeat for h steps total
  4. Accumulate danger:
     danger(a) = Σ_t γ^t · occupancy[t][ego_pos_t(a)]
```

All N×5 candidate trajectories are batched per step.

#### Step 5: Adjust Probabilities

```
P_adjusted(a) ∝ P_policy(a) · exp(-λ · danger(a))
```

- High `safety_lambda` → more conservative (stronger avoidance)
- `gamma` controls how much future danger matters vs. immediate

#### Batching Strategy

All forward passes are batched for efficiency:

| Pass | Batch size | Count |
|------|-----------|-------|
| Ego action probs (full context) | N | 1 |
| Neighbor trajectory simulation | # unique hp neighbors | h |
| Ego candidate rollout | N × 5 | h − 1 |
| **Total forward passes** | | **2h** |

For N=32 agents, h=3: **6 batched forward passes per timestep**.

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
| `self.sim_encoder` | `sim_num_agents` (default 1) | Simulating neighbors + ego rollout — reduced window |

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

### Tuning multi-step parameters

```bash
python example_safe.py \
  --model 2M \
  --num_agents 64 \
  --device mps \
  --horizon 5 \
  --gamma 0.85 \
  --safety_lambda 10.0 \
  --priority_scheme random
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
    gamma=0.9,
    safety_lambda=5.0,
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
| `horizon` | `3` | Steps to simulate forward. 1 = single-step hard mask. >1 = multi-step cost field. |
| `gamma` | `0.9` | Discount factor for future occupancy danger (0 < gamma <= 1). |
| `safety_lambda` | `5.0` | Penalty strength. `P(a) ∝ P_policy(a) · exp(-λ · danger(a))`. Higher = more conservative. |
| `do_sample` | `True` | Sample from adjusted distribution vs. argmax. |

## Limitations and Future Work

1. **Approximate neighbor simulation**: When ego simulates neighbor j, it can only
   include agents that ego itself can see. Neighbor j might see additional agents
   outside ego's radius. This is a fundamental decentralized constraint.

2. **Compounding prediction error**: Each simulation step builds on the previous
   step's predicted position. By step h, the predicted trajectory may diverge from
   reality. In practice h=3–5 is a sweet spot.

3. **Argmax trajectory, soft occupancy**: Neighbor trajectories advance by argmax
   (most likely action), but the occupancy field records the full distribution at
   each step. A tree-based approach (branching on all actions) would be more
   accurate but exponentially more expensive.

4. **Same-priority conflicts**: Agents only yield to strictly higher-priority
   neighbors. Two agents with adjacent priorities that can't see each other may
   still collide. This is inherent to the decentralized setting.

5. **Static other-agent positions during simulation**: When simulating a neighbor
   forward, other agents' positions are not updated (they stay at their t=0
   positions). This is an approximation that could be improved with joint
   simulation, at the cost of complexity.

## Git History

Branch: `feat/safe-action-wrapper`

```
3797d9e feat: Add multi-step probabilistic occupancy cost field
fb638a9 refactor: Make all agents commit simultaneously, priority as tie-breaker only
7d24a7c feat: Rewrite as fully decentralized wrapper with fixed priorities
bbc9d5e feat: Add SafeActionWrapper for conflict-free decentralized action selection
```
