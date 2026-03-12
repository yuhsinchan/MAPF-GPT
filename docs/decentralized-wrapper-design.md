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
├── __init__(cfg, priority_scheme, sim_num_agents)
├── act(observations) → List[int]           # main entry point
├── act_with_info(observations) → dict      # with diagnostics
├── get_action_probs(observations) → (N,5)  # raw policy probs
├── simulate_neighbor(neighbor, ego, obs)   # single neighbor
├── simulate_visible_neighbors(ego, obs)    # batch neighbors
└── get_safe_action(obs, penalty, sample)   # core safety logic
```

## How It Works

### Assumptions

1. All agents share the **same policy network** (same weights).
2. All agents know visible neighbors' **goals and IDs**.
3. A **fixed priority** ordering is assigned at episode start and known to all.
4. Only agents within **observation radius** are relevant for conflict checks —
   distant agents are ignored.

### Algorithm (per timestep)

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

The reduced window makes neighbor simulation cheaper (fewer tokens to process)
while retaining the cost2go spatial context. The `sim_num_agents` parameter
controls this tradeoff.

### Batching Strategy

All forward passes are batched for efficiency:

1. **Ego batch**: All N agents' full-context observations → single forward pass → `(N, 5)` probs.
2. **Simulation batch**: All neighbor simulations across all agents are collected
   into one list and run in a single forward pass. For N agents with ~k
   higher-priority neighbors each, this is ~Nk/2 observations in one batch.

## Usage

### Basic (drop-in)

```bash
python example_safe.py \
  --map_name validation-mazes-seed-000 \
  --model 2M \
  --num_agents 32 \
  --device mps
```

### With options

```bash
python example_safe.py \
  --model 2M \
  --num_agents 32 \
  --device mps \
  --priority_scheme random \
  --sim_num_agents 5
```

### Programmatic

```python
from gpt.inference import MAPFGPTInferenceConfig
from gpt.safe_action_wrapper import DecentralizedWrapper

cfg = MAPFGPTInferenceConfig(path_to_weights="weights/model-2M.pt", device="mps")
wrapper = DecentralizedWrapper(cfg, priority_scheme="index", sim_num_agents=1)
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
# Get raw action probabilities for all agents
probs = wrapper.get_action_probs(observations)  # (N, 5)

# Simulate a specific neighbor from ego's perspective
neighbor_probs = wrapper.simulate_neighbor(
    neighbor_idx=3, ego_idx=0, observations=observations
)  # (5,)

# Simulate all higher-priority visible neighbors of agent 5
neighbor_dict = wrapper.simulate_visible_neighbors(
    ego_idx=5, observations=observations, only_higher_priority=True
)  # {neighbor_idx: (5,) tensor}
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `priority_scheme` | `"index"` | `"index"`: agent 0 = highest priority. `"random"`: shuffled at episode start. |
| `sim_num_agents` | `1` | Agent slots in reduced window for neighbor simulation. 1 = only the simulated agent itself. Higher = richer context but slower. |
| `conflict_penalty` | `0.0` | Multiplier for conflicting action probs. 0.0 = hard mask. >0 = soft penalty. |
| `do_sample` | `True` | Sample from adjusted distribution vs. argmax. |

## Limitations and Future Work

1. **Approximate neighbor simulation**: When ego simulates neighbor j, it can only
   include agents that ego itself can see. Neighbor j might see additional agents
   outside ego's radius. This is a fundamental decentralized constraint.

2. **Single-step lookahead**: Only checks conflicts one step ahead. Multi-step
   rollout (simulating k steps) would catch more conflicts but costs k× more
   forward passes.

3. **Argmax prediction**: Neighbors' actions are predicted as argmax of their
   policy distribution. A probabilistic version could weight conflict penalties
   by the probability of each neighbor action, but adds complexity.

4. **Same-priority conflicts**: Currently agents only yield to strictly
   higher-priority neighbors. Two agents with adjacent priorities that can't see
   each other may still collide. This is inherent to the decentralized setting.

## Git History

Branch: `feat/safe-action-wrapper`

```
fb638a9 refactor: Make all agents commit simultaneously, priority as tie-breaker only
7d24a7c feat: Rewrite as fully decentralized wrapper with fixed priorities
bbc9d5e feat: Add SafeActionWrapper for conflict-free decentralized action selection
```
