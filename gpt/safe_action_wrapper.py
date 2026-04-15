"""
Decentralized SafeActionWrapper for MAPF-GPT agents.

Fully decentralized: each agent acts independently using only local observations.
All agents share the same policy network, so each can forward-simulate what
its neighbors will do — no communication required.

Key assumptions:
- All agents know visible neighbors' goals and IDs.
- Fixed priority ordering assigned at episode start (lower index = higher priority).
- Only nearby agents (within observation radius) are checked for conflicts.
- Higher-priority agents' predicted actions are treated as committed;
  lower-priority agents must yield.
"""

import heapq
import math
from typing import Dict, List, Literal, Optional

import cppimport.import_hook
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from pathlib import Path

from gpt.model import GPT, GPTConfig
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig, strip_prefix_from_state_dict
from tokenizer import cost2go
from tokenizer.tokenizer import Encoder, InputParameters

# Action index <-> displacement mapping (matches POGEMA conventions)
ACTION_TO_DELTA = {
    0: (0, 0),   # wait
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}

MOVES_STR = {0: "w", 1: "u", 2: "d", 3: "l", 4: "r"}


def count_collisions(positions: list, actions: list) -> dict:
    """
    Count vertex and edge collisions from positions and actions.

    Args:
        positions: List of (x, y) tuples, one per agent.
        actions: List of action indices (0-4), one per agent.

    Returns:
        Dict with:
            vertex: number of vertex collisions (2+ agents at same cell)
            edge: number of edge collisions (two agents swapping cells)
            total: vertex + edge
            agents_in_vertex: set of agent indices involved in vertex collisions
            agents_in_edge: set of agent indices involved in edge collisions
    """
    n = len(positions)
    next_positions = []
    for i in range(n):
        dx, dy = ACTION_TO_DELTA[actions[i]]
        next_positions.append((positions[i][0] + dx, positions[i][1] + dy))

    vertex_collisions = 0
    edge_collisions = 0
    agents_in_vertex = set()
    agents_in_edge = set()

    # Vertex: multiple agents at the same next position
    cell_to_agents: Dict[tuple, List[int]] = {}
    for i, pos in enumerate(next_positions):
        cell_to_agents.setdefault(pos, []).append(i)
    for cell, agents_at_cell in cell_to_agents.items():
        if len(agents_at_cell) > 1:
            vertex_collisions += len(agents_at_cell) - 1
            agents_in_vertex.update(agents_at_cell)

    # Edge: agents swap positions
    for i in range(n):
        for j in range(i + 1, n):
            if next_positions[i] == positions[j] and next_positions[j] == positions[i]:
                edge_collisions += 1
                agents_in_edge.update([i, j])

    return {
        "vertex": vertex_collisions,
        "edge": edge_collisions,
        "total": vertex_collisions + edge_collisions,
        "agents_in_vertex": agents_in_vertex,
        "agents_in_edge": agents_in_edge,
    }


def _load_model(cfg: MAPFGPTInferenceConfig) -> GPT:
    """Download weights if needed, load model onto device."""
    path_to_weights = Path(cfg.path_to_weights)
    if path_to_weights.name in ['model-2M.pt', 'model-6M.pt', 'model-85M.pt']:
        hf_hub_download(
            repo_id=cfg.repo_id,
            filename=path_to_weights.name,
            local_dir=path_to_weights.parent,
        )

    if cfg.device == 'cuda' and not torch.cuda.is_available():
        cfg.device = 'cpu'
    elif cfg.device == 'mps' and not torch.backends.mps.is_available():
        cfg.device = 'cpu'

    checkpoint = torch.load(path_to_weights, map_location=cfg.device)
    model_state_dict = strip_prefix_from_state_dict(checkpoint["model"])
    gpt_config = GPTConfig(**checkpoint.get("model_args"))
    net = GPT(gpt_config)
    net.load_state_dict(model_state_dict, strict=False)
    net.to(cfg.device)
    net.eval()
    return net


def _make_encoder(cfg: MAPFGPTInferenceConfig, num_agents: int) -> Encoder:
    """Create an Encoder with the given num_agents slot count."""
    return Encoder(InputParameters(
        num_agents=num_agents,
        num_previous_actions=cfg.num_previous_actions,
        cost2go_value_limit=cfg.cost2go_value_limit,
        agents_radius=cfg.agents_radius,
        cost2go_radius=cfg.cost2go_radius,
        context_size=cfg.context_size,
        mask_actions_history=cfg.mask_actions_history,
        mask_cost2go=cfg.mask_cost2go,
        mask_goal=cfg.mask_goal,
        mask_greed_action=cfg.mask_greed_action,
    ))


def _apply_pos(pos: tuple, action: int) -> tuple:
    """Apply an action to a position, returning the new position."""
    dx, dy = ACTION_TO_DELTA[action]
    return (pos[0] + dx, pos[1] + dy)


class DecentralizedWrapper:
    """
    Fully decentralized safety wrapper with fixed priorities and
    risk map-based uncertainty-aware planning.

    Each agent independently:
    1. Runs the shared policy to get its own action distribution.
    2. Simulates higher-priority visible neighbors' trajectory trees,
       building a cumulative occupancy risk map.
    3. Propagates risk along distance-to-goal paths.
    4. Selects actions that balance policy preference against collision risk.

    Usage (drop-in replacement for MAPFGPTInference):
        wrapper = DecentralizedWrapper(cfg)
        wrapper.reset_states()
        actions = wrapper.act(observations)
    """

    def __init__(
        self,
        cfg: MAPFGPTInferenceConfig,
        priority_scheme: Literal["index", "random"] = "index",
        sim_num_agents: int = 1,
        horizon: int = 1,
        epsilon: float = 0.1,
        alpha: float = 0.5,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        sequential_simulation: bool = False,
        conflict_radius: Optional[int] = None,
    ):
        """
        Args:
            cfg: Standard MAPF-GPT inference config.
            priority_scheme: How to assign priorities.
                "index" = agent 0 has highest priority.
                "random" = random shuffle at episode start.
            sim_num_agents: Number of agent slots in reduced context window
                for neighbor forward simulation.
            horizon: Number of steps to simulate forward. 1 = single-step
                hard-mask (original behavior). >1 = risk map approach.
            epsilon: Pruning threshold for trajectory tree. Actions with
                probability below epsilon are dropped and remaining
                probabilities renormalized (Eq. 1).
            alpha: Decay factor for implicit risk propagation along
                distance-to-goal paths (0 < alpha < 1) (Eq. 4).
            lambda_1: Weight for policy log-probability in cost function.
                Higher = prefer policy-recommended actions (Eq. 5).
            lambda_2: Weight for risk map in cost function.
                Higher = more conservative collision avoidance (Eq. 5).
            sequential_simulation: If True, simulate hp neighbors in
                priority order where each agent avoids occupancy from
                higher-priority agents. If False, simulate independently.
            conflict_radius: Chebyshev distance threshold for trajectory
                tree simulation. Only hp neighbors within this radius
                trigger a tree simulation. Agents between conflict_radius
                and agents_radius are still included in the ego's context
                window for action probability computation, but do not
                generate forward passes. None = use agents_radius (no filter).
                Smaller values reduce forward passes in sparse scenarios.
        """
        self.cfg = cfg
        self.priority_scheme = priority_scheme
        self.sim_num_agents = sim_num_agents
        self.horizon = horizon
        self.epsilon = epsilon
        self.alpha = alpha
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.sequential_simulation = sequential_simulation
        self.conflict_radius = conflict_radius if conflict_radius is not None else cfg.agents_radius

        self.net = _load_model(cfg)
        self.encoder = _make_encoder(cfg, cfg.num_agents)
        self.sim_encoder = _make_encoder(cfg, sim_num_agents)

        # Per-agent state (initialized on first observation)
        self.num_agents: Optional[int] = None
        self.priorities: Optional[List[int]] = None
        self.cost2go_data = None
        self.action_histories: Optional[List[List[str]]] = None
        self.position_histories: Optional[List[List]] = None

    def reset_states(self):
        self.num_agents = None
        self.priorities = None
        self.cost2go_data = None
        self.action_histories = None
        self.position_histories = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _initialize(self, observations):
        n = len(observations)
        self.num_agents = n

        if self.priority_scheme == "random":
            import random
            self.priorities = list(range(n))
            random.shuffle(self.priorities)
        else:
            self.priorities = list(range(n))

        global_obs = observations[0]["global_obstacles"].copy().astype(int).tolist()
        self.cost2go_data = cost2go.precompute_cost2go(
            global_obs, self.cfg.cost2go_radius
        )

        self.action_histories = [
            ["n"] * self.cfg.num_previous_actions for _ in range(n)
        ]
        self.position_histories = [[obs["global_xy"]] for obs in observations]

    def _update_histories(self, observations):
        moves = {(0, 0): "w", (-1, 0): "u", (1, 0): "d", (0, -1): "l", (0, 1): "r"}
        for i in range(self.num_agents):
            self.position_histories[i].append(observations[i]["global_xy"])
            dx = self.position_histories[i][-1][0] - self.position_histories[i][-2][0]
            dy = self.position_histories[i][-1][1] - self.position_histories[i][-2][1]
            self.action_histories[i].append(moves[(dx, dy)])
            self.action_histories[i] = self.action_histories[i][
                -self.cfg.num_previous_actions:
            ]

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _compute_greedy_action(self, pos, target) -> str:
        """Compute the 4-bit greedy action string for the tokenizer."""
        result = ""
        for m in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            new_pos = (pos[0] + m[0], pos[1] + m[1])
            if (
                self.cost2go_data[target][new_pos[0]][new_pos[1]] >= 0
                and self.cost2go_data[target][pos[0]][pos[1]]
                > self.cost2go_data[target][new_pos[0]][new_pos[1]]
            ):
                result += "1"
            else:
                result += "0"
        return result

    def _get_visible_neighbors(self, ego_idx: int, observations) -> List[int]:
        """Return indices of agents within agents_radius of ego."""
        ego_pos = observations[ego_idx]["global_xy"]
        r = self.cfg.agents_radius
        neighbors = []
        for i in range(self.num_agents):
            if i == ego_idx:
                continue
            other_pos = observations[i]["global_xy"]
            if (
                abs(other_pos[0] - ego_pos[0]) <= r
                and abs(other_pos[1] - ego_pos[1]) <= r
            ):
                neighbors.append(i)
        return neighbors

    def _get_sorted_context_agents(
        self, ego_idx: int, observations, candidate_pool: Optional[List[int]] = None
    ) -> List[int]:
        """Get agents sorted by cost2go distance from ego."""
        ego_pos = tuple(observations[ego_idx]["global_xy"])
        pool = candidate_pool if candidate_pool is not None else list(range(self.num_agents))
        distances = []
        for j in pool:
            pos_j = observations[j]["global_xy"]
            d = self.cost2go_data[ego_pos][pos_j[0]][pos_j[1]]
            if d >= 0:
                distances.append((j, d))
        distances.sort(key=lambda x: (x[1], x[0]))
        return [agent_id for agent_id, _ in distances]

    def _build_input(
        self, ego_idx: int, observations, max_agents: int, context_agents: List[int]
    ) -> dict:
        """Build tokenizer input dict for a single agent from observations."""
        ego_pos = observations[ego_idx]["global_xy"]
        ego_target = tuple(observations[ego_idx]["global_target_xy"])

        agents_info = []
        for n in context_agents[:max_agents]:
            n_pos = observations[n]["global_xy"]
            n_target = observations[n]["global_target_xy"]
            relative_xy = (n_pos[0] - ego_pos[0], n_pos[1] - ego_pos[1])
            relative_goal = (n_target[0] - ego_pos[0], n_target[1] - ego_pos[1])

            if (
                -self.cfg.agents_radius <= relative_xy[0] <= self.cfg.agents_radius
                and -self.cfg.agents_radius <= relative_xy[1] <= self.cfg.agents_radius
            ):
                agents_info.append({
                    "relative_pos": relative_xy,
                    "relative_goal": relative_goal,
                    "previous_actions": self.action_histories[n],
                    "next_action": self._compute_greedy_action(
                        tuple(n_pos), tuple(n_target)
                    ),
                })

        return {
            "agents": agents_info,
            "cost2go": cost2go.generate_cost2go_obs(
                self.cost2go_data[ego_target],
                observations[ego_idx]["global_xy"],
                self.cfg.cost2go_radius,
                self.cfg.cost2go_value_limit,
                self.cfg.mask_cost2go,
            ),
        }

    def _build_sim_input(self, pos: tuple, target: tuple, action_history: list) -> dict:
        """
        Build tokenizer input for a single agent at a simulated position.

        Used for multi-step forward simulation where the agent's position
        has been hypothetically advanced. No context agents — only the
        cost2go grid provides spatial information.
        """
        agents_info = [{
            "relative_pos": (0, 0),
            "relative_goal": (target[0] - pos[0], target[1] - pos[1]),
            "previous_actions": action_history,
            "next_action": self._compute_greedy_action(pos, target),
        }]
        return {
            "agents": agents_info,
            "cost2go": cost2go.generate_cost2go_obs(
                self.cost2go_data[target],
                [pos[0], pos[1]],
                self.cfg.cost2go_radius,
                self.cfg.cost2go_value_limit,
                self.cfg.mask_cost2go,
            ),
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_batch(self, inputs: list, encoder: Encoder) -> torch.Tensor:
        """
        Run the policy on a batch of tokenizer inputs.

        Returns:
            probs: Tensor of shape (batch, 5).
        """
        if not inputs:
            return torch.empty(0, 5, device=self.cfg.device)

        tensor_obs = torch.tensor(
            [encoder.encode(inp) for inp in inputs],
            dtype=torch.long,
            device=self.cfg.device,
        )
        with torch.no_grad():
            logits, _ = self.net(tensor_obs)
            logits = logits[:, -1, :]
            masked = torch.full_like(logits, float("-inf"))
            masked[:, :5] = logits[:, :5]
            probs = F.softmax(masked, dim=-1)[:, :5]
        return probs

    # ------------------------------------------------------------------
    # Trajectory tree simulation (Eq. 1-2)
    # ------------------------------------------------------------------

    def _prune_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Prune actions below epsilon and renormalize (Eq. 1).

        Actions with probability < epsilon are zeroed out.
        Remaining probabilities are renormalized to sum to 1.
        """
        mask = probs >= self.epsilon
        pruned = probs * mask
        total = pruned.sum()
        if total > 0:
            return pruned / total
        return probs  # fallback: no pruning if all below epsilon

    def _prune_probs_batch(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Batched version of _prune_probs for shape (N, 5).

        Avoids per-row Python loops — the entire batch is pruned and
        renormalized in two tensor ops.
        """
        mask = probs >= self.epsilon
        pruned = probs * mask
        totals = pruned.sum(dim=-1, keepdim=True)
        # Where everything was pruned, fall back to the original distribution
        fallback = totals == 0
        pruned = torch.where(fallback.expand_as(pruned), probs, pruned)
        totals = pruned.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return pruned / totals

    def _collect_hp_neighbors(self, observations) -> set:
        """
        Collect all unique higher-priority neighbors that are close enough
        to warrant trajectory tree simulation.

        Uses conflict_radius (≤ agents_radius) rather than the full
        agents_radius so that distant-but-visible hp neighbors don't
        generate unnecessary forward passes. The ego's context window
        (for its own action probs) still uses the full agents_radius.
        """
        hp_neighbors = set()
        r = self.conflict_radius
        for ego_idx in range(self.num_agents):
            ego_pos = observations[ego_idx]["global_xy"]
            for n in range(self.num_agents):
                if n == ego_idx:
                    continue
                if self.priorities[n] >= self.priorities[ego_idx]:
                    continue
                other_pos = observations[n]["global_xy"]
                if (
                    abs(other_pos[0] - ego_pos[0]) <= r
                    and abs(other_pos[1] - ego_pos[1]) <= r
                ):
                    hp_neighbors.add(n)
        return hp_neighbors

    def _simulate_single_agent_tree(
        self,
        start_pos: tuple,
        target: tuple,
        start_history: list,
        horizon: int,
        prior_occupancy: Optional[List[Dict[tuple, float]]] = None,
    ) -> List[Dict[tuple, float]]:
        """
        Simulate one agent's trajectory tree for h steps (Eq. 2).

        Used only in sequential mode (prior_occupancy != None).
        In independent mode, _simulate_hp_trajectory_trees handles all
        agents together in one batched forward pass per step.

        Args:
            start_pos: Agent's starting position.
            target: Agent's goal position.
            start_history: Agent's action history at t=0.
            horizon: Number of steps to simulate.
            prior_occupancy: If provided (sequential mode), occupancy from
                higher-priority agents. Actions landing on high-occupancy
                cells are penalized before tree propagation.

        Returns:
            List of h occupancy dicts. occupancy[t][(x,y)] = probability mass.
        """
        # Frontier: pos -> (mass, history)
        frontier: Dict[tuple, tuple] = {
            start_pos: (1.0, list(start_history))
        }
        occupancy_per_step: List[Dict[tuple, float]] = []

        for t in range(horizon):
            frontier_items = list(frontier.items())
            if not frontier_items:
                occupancy_per_step.append({})
                continue

            # Build inputs for all frontier positions
            inputs = [
                self._build_sim_input(pos, target, hist)
                for pos, (mass, hist) in frontier_items
            ]

            # One batched forward pass for this agent's frontier
            raw_probs = self._forward_batch(inputs, self.sim_encoder)

            # In sequential mode, apply prior-occupancy penalty before pruning
            if prior_occupancy is not None and t < len(prior_occupancy):
                adjusted = raw_probs.clone()
                for idx, (pos, (mass, hist)) in enumerate(frontier_items):
                    for a in range(5):
                        occ_risk = prior_occupancy[t].get(_apply_pos(pos, a), 0.0)
                        if occ_risk > 0:
                            adjusted[idx, a] *= (1.0 - occ_risk)
                row_sums = adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                raw_probs = adjusted / row_sums

            # Prune and renormalize the entire batch at once, then transfer to
            # numpy once to avoid per-element .item() overhead
            all_pruned_np = self._prune_probs_batch(raw_probs).cpu().numpy()

            new_frontier: Dict[tuple, tuple] = {}
            step_occ: Dict[tuple, float] = {}

            for idx, (pos, (mass, hist)) in enumerate(frontier_items):
                pruned = all_pruned_np[idx]
                for a in range(5):
                    p = float(pruned[a])
                    if p < 1e-8:
                        continue
                    next_pos = _apply_pos(pos, a)
                    transition_mass = mass * p
                    step_occ[next_pos] = step_occ.get(next_pos, 0.0) + transition_mass

                    # Update frontier: keep history from highest-mass path
                    new_hist = (hist + [MOVES_STR[a]])[-self.cfg.num_previous_actions:]
                    if next_pos in new_frontier:
                        existing_mass, _ = new_frontier[next_pos]
                        new_frontier[next_pos] = (
                            existing_mass + transition_mass,
                            new_hist if transition_mass > existing_mass else new_frontier[next_pos][1],
                        )
                    else:
                        new_frontier[next_pos] = (transition_mass, new_hist)

            occupancy_per_step.append(step_occ)
            frontier = new_frontier

        return occupancy_per_step

    def _simulate_hp_trajectory_trees(
        self, observations, horizon: int
    ) -> Dict[int, List[Dict[tuple, float]]]:
        """
        Simulate all unique higher-priority neighbors' trajectory trees.

        Independent mode: all agents' frontiers are batched into ONE forward
        pass per horizon step (h+1 total passes including the ego pass).
        This is the primary performance optimization vs. the naive K*h passes.

        Sequential mode: simulate in priority order using _simulate_single_agent_tree
        so each agent can penalize actions that land on higher-priority occupancy.

        Returns:
            Dict[neighbor_idx -> list of h occupancy dicts].
        """
        all_hp_neighbors = self._collect_hp_neighbors(observations)
        if not all_hp_neighbors:
            return {}

        hp_list = sorted(all_hp_neighbors, key=lambda n: self.priorities[n])
        trajectories: Dict[int, List[Dict[tuple, float]]] = {}

        if not self.sequential_simulation:
            # ---- Independent mode: one forward pass per horizon step ----
            # Initialize a frontier per hp agent: pos -> (mass, history)
            frontiers: Dict[int, Dict[tuple, tuple]] = {}
            agent_targets: Dict[int, tuple] = {}
            for n in hp_list:
                pos = tuple(observations[n]["global_xy"])
                target = tuple(observations[n]["global_target_xy"])
                frontiers[n] = {pos: (1.0, list(self.action_histories[n]))}
                agent_targets[n] = target
                trajectories[n] = []

            for t in range(horizon):
                # Collect ALL frontier cells from ALL agents into one batch
                batch_inputs = []
                batch_meta: List[tuple] = []  # (agent_n, pos, mass, hist)
                for n in hp_list:
                    target = agent_targets[n]
                    for pos, (mass, hist) in frontiers[n].items():
                        batch_inputs.append(self._build_sim_input(pos, target, hist))
                        batch_meta.append((n, pos, mass, hist))

                if not batch_inputs:
                    for n in hp_list:
                        trajectories[n].append({})
                    continue

                # ONE forward pass for all agents' frontiers at this step
                raw_probs = self._forward_batch(batch_inputs, self.sim_encoder)
                # Prune entire batch + transfer to numpy once (avoids .item() overhead)
                all_pruned_np = self._prune_probs_batch(raw_probs).cpu().numpy()

                new_frontiers: Dict[int, Dict[tuple, tuple]] = {n: {} for n in hp_list}
                step_occs: Dict[int, Dict[tuple, float]] = {n: {} for n in hp_list}

                for idx, (n, pos, mass, hist) in enumerate(batch_meta):
                    pruned = all_pruned_np[idx]
                    step_occ = step_occs[n]
                    new_frontier = new_frontiers[n]
                    for a in range(5):
                        p = float(pruned[a])
                        if p < 1e-8:
                            continue
                        next_pos = _apply_pos(pos, a)
                        transition_mass = mass * p
                        step_occ[next_pos] = step_occ.get(next_pos, 0.0) + transition_mass
                        new_hist = (hist + [MOVES_STR[a]])[-self.cfg.num_previous_actions:]
                        if next_pos in new_frontier:
                            ex_mass, ex_hist = new_frontier[next_pos]
                            new_frontier[next_pos] = (
                                ex_mass + transition_mass,
                                new_hist if transition_mass > ex_mass else ex_hist,
                            )
                        else:
                            new_frontier[next_pos] = (transition_mass, new_hist)

                for n in hp_list:
                    trajectories[n].append(step_occs[n])
                    frontiers[n] = new_frontiers[n]

            return trajectories

        else:
            # Sequential mode: simulate in priority order
            # Each agent sees the cumulative occupancy from higher-priority agents
            cumulative_occupancy: List[Dict[tuple, float]] = [
                {} for _ in range(horizon)
            ]
            for n in hp_list:
                pos = tuple(observations[n]["global_xy"])
                target = tuple(observations[n]["global_target_xy"])
                history = list(self.action_histories[n])
                agent_occ = self._simulate_single_agent_tree(
                    pos, target, history, horizon,
                    prior_occupancy=cumulative_occupancy,
                )
                trajectories[n] = agent_occ
                # Update cumulative occupancy with this agent's contribution
                for t in range(horizon):
                    for cell, prob in agent_occ[t].items():
                        cumulative_occupancy[t][cell] = max(
                            cumulative_occupancy[t].get(cell, 0.0), prob
                        )

        return trajectories

    # ------------------------------------------------------------------
    # Risk map construction (Eq. 3-4)
    # ------------------------------------------------------------------

    def _propagate_risk(
        self, risk_map: Dict[tuple, float], target: tuple
    ) -> Dict[tuple, float]:
        """
        Propagate risk along distance-to-goal paths with decay alpha (Eq. 4).

        For each risky cell (i,j), spread risk to cells (x,y) that are
        closer to the goal: R[x,y] <- max(R[x,y], R[i,j] * alpha^(d_ij - d_xy))

        Uses a max-heap (by distance-to-goal) so cells are processed from
        farthest to closest. When a neighbor's risk is updated it is pushed
        back onto the heap, giving transitive propagation in one pass without
        re-iterating the full cell list.
        """
        d = self.cost2go_data[target]
        propagated = dict(risk_map)

        # Seed heap with all risky cells.  Heap stores (-dist, i, j) so the
        # cell farthest from the goal is always popped first.
        heap = []
        for (i, j), risk in risk_map.items():
            dist = d[i][j]
            if dist >= 0 and risk > 1e-8:
                heapq.heappush(heap, (-dist, i, j))

        while heap:
            neg_d_ij, i, j = heapq.heappop(heap)
            d_ij = -neg_d_ij

            # A cell may have been pushed multiple times; use the current
            # (highest) risk value rather than the stale one from when it
            # was pushed.
            risk = propagated.get((i, j), 0.0)
            if risk < 1e-8:
                continue

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x, y = i + dx, j + dy
                try:
                    d_xy = d[x][y]
                except IndexError:
                    continue
                if d_xy < 0 or d_xy >= d_ij:
                    continue
                propagated_risk = risk * (self.alpha ** (d_ij - d_xy))
                if propagated_risk > 1e-8 and propagated_risk > propagated.get((x, y), 0.0):
                    propagated[(x, y)] = propagated_risk
                    heapq.heappush(heap, (-d_xy, x, y))

        return propagated

    def _build_cumulative_risk_map(
        self, ego_idx: int, observations, trajectories: dict
    ) -> Dict[int, Dict[tuple, float]]:
        """
        Build per-ego cumulative risk map from higher-priority neighbors (Eq. 3).

        For each hp neighbor visible to ego:
        1. Keep occupancy per timestep (not collapsed).
        2. Propagate each timestep's occupancy along the neighbor's goal direction.
        3. Aggregate across neighbors using element-wise max per timestep.

        Returns:
            Dict[timestep -> Dict[(x,y) -> risk]], timestep is 1-indexed (1..h).
            Preserves the temporal dimension so the cost function can check risk
            at the exact timestep the ego would arrive at a destination cell.
        """
        visible = self._get_visible_neighbors(ego_idx, observations)
        hp_neighbors = [
            n for n in visible
            if self.priorities[n] < self.priorities[ego_idx] and n in trajectories
        ]

        if not hp_neighbors:
            return {}

        cumulative_risk: Dict[int, Dict[tuple, float]] = {}

        for n in hp_neighbors:
            n_target = tuple(observations[n]["global_target_xy"])
            for t_idx, step_occ in enumerate(trajectories[n]):
                t = t_idx + 1  # 1-indexed: step 1 = ego's immediate next position
                # Propagate this timestep's occupancy along the neighbor's goal path
                propagated = self._propagate_risk(step_occ, n_target)
                # Aggregate across neighbors with element-wise max (Eq. 3)
                t_risk = cumulative_risk.setdefault(t, {})
                for cell, risk in propagated.items():
                    t_risk[cell] = max(t_risk.get(cell, 0.0), risk)

        return cumulative_risk

    # ------------------------------------------------------------------
    # Single-step safety (unchanged)
    # ------------------------------------------------------------------

    def _predict_hp_next_positions(
        self, observations, positions: list
    ) -> Dict[int, Dict[int, tuple]]:
        """
        Predict next positions of higher-priority neighbors for each ego.

        Returns:
            Dict[ego_idx -> Dict[neighbor_idx -> predicted_next_pos]].
        """
        sim_requests = []
        sim_inputs = []
        for ego_idx in range(self.num_agents):
            visible = self._get_visible_neighbors(ego_idx, observations)
            higher = [
                n for n in visible
                if self.priorities[n] < self.priorities[ego_idx]
            ]
            if not higher:
                continue
            ego_visible_set = set(visible) | {ego_idx}
            for n in higher:
                pool = list(ego_visible_set | {n})
                context = self._get_sorted_context_agents(n, observations, pool)
                sim_inputs.append(
                    self._build_input(n, observations, self.sim_num_agents, context)
                )
                sim_requests.append((ego_idx, n))

        sim_probs = self._forward_batch(sim_inputs, self.sim_encoder)

        hp_predicted: Dict[int, Dict[int, tuple]] = {}
        for i, (ego_idx, n) in enumerate(sim_requests):
            predicted_action = torch.argmax(sim_probs[i]).item()
            n_next = _apply_pos(positions[n], predicted_action)
            hp_predicted.setdefault(ego_idx, {})[n] = n_next

        return hp_predicted

    def _mask_conflicts(
        self,
        ego_pos: tuple,
        ego_probs: torch.Tensor,
        neighbors_next: Dict[int, tuple],
        positions: list,
    ) -> torch.Tensor:
        """
        Zero out actions that cause vertex or edge conflicts with
        higher-priority neighbors' predicted next positions.
        """
        masked = ego_probs.clone()
        for action_idx in range(5):
            ego_next = _apply_pos(ego_pos, action_idx)
            for other_idx, other_next in neighbors_next.items():
                other_pos = positions[other_idx]
                # Vertex conflict
                if ego_next == other_next:
                    masked[action_idx] = 0.0
                    break
                # Edge conflict (swap)
                if ego_next == other_pos and other_next == ego_pos:
                    masked[action_idx] = 0.0
                    break
        return masked

    def _sample_action(
        self, probs: torch.Tensor, do_sample: bool
    ) -> int:
        """Sample or argmax from a probability distribution. Falls back to wait."""
        total = probs.sum()
        if total == 0:
            return 0  # wait
        if do_sample:
            return torch.multinomial(probs / total, num_samples=1).item()
        return torch.argmax(probs).item()

    def _get_safe_action_single_step(
        self, observations, do_sample: bool
    ) -> List[int]:
        """Single-step hard-mask: zero out conflicting actions, then sample."""
        positions = [tuple(obs["global_xy"]) for obs in observations]
        all_probs = self.get_action_probs(observations)
        hp_predicted = self._predict_hp_next_positions(observations, positions)

        final_actions = []
        for ego_idx in range(self.num_agents):
            ego_probs = self._mask_conflicts(
                positions[ego_idx],
                all_probs[ego_idx],
                hp_predicted.get(ego_idx, {}),
                positions,
            )
            final_actions.append(self._sample_action(ego_probs, do_sample))

        return final_actions

    # ------------------------------------------------------------------
    # Multi-step risk map approach (Eq. 5-6)
    # ------------------------------------------------------------------

    def _get_safe_action_multistep(
        self, observations, do_sample: bool
    ) -> List[int]:
        """
        Risk map-based action selection.

        1. Get ego action probs (full context) — 1 forward pass.
        2. Simulate hp neighbors' trajectory trees — h forward passes.
        3. Build per-ego cumulative risk map with propagation.
        4. Select actions via cost function: c(a) = -lambda_1 * log pi(a) + lambda_2 * R[dest(a)]
        """
        positions = [tuple(obs["global_xy"]) for obs in observations]

        # 1. Ego action probs (full context)
        all_probs = self.get_action_probs(observations)

        # 2. Simulate hp neighbors' trajectory trees
        trajectories = self._simulate_hp_trajectory_trees(
            observations, self.horizon
        )

        # 3. Build per-ego cumulative risk maps
        ego_risk_maps = []
        for ego_idx in range(self.num_agents):
            if trajectories:
                risk = self._build_cumulative_risk_map(
                    ego_idx, observations, trajectories
                )
            else:
                risk = {}
            ego_risk_maps.append(risk)

        # 4. Action selection via cost function (Eq. 5-6)
        # risk[t] is the risk map for timestep t (1-indexed).
        # The ego takes one action and arrives at next_pos at t=1, so we
        # check risk at t=1 only — avoids penalizing cells that are only
        # risky at future timesteps the ego won't reach in one step.
        #
        # Costs are converted to a sampling distribution via softmin:
        #   weight(a) = exp(-c(a)) = π(a)^λ₁ · exp(-λ₂ · R[dest(a)])
        # When do_sample=True this distribution is sampled, giving the same
        # stochasticity as the baseline policy while still down-weighting
        # risky actions. When do_sample=False the minimum-cost action is taken.
        #
        # Vectorized: convert all_probs to numpy once, do dict lookups and
        # math in Python/numpy, build the weight tensor in one shot.
        all_probs_np = all_probs.cpu().numpy()  # (N, 5) — one transfer for all agents
        log_probs_np = -np.log(np.clip(all_probs_np, 1e-8, None))  # (N, 5)

        final_actions = []
        for ego_idx in range(self.num_agents):
            ego_pos = positions[ego_idx]
            risk_t1 = ego_risk_maps[ego_idx].get(1, {})
            p_row = all_probs_np[ego_idx]   # (5,) numpy
            lp_row = log_probs_np[ego_idx]  # (5,) numpy

            weights = torch.zeros(5)
            for a in range(5):
                if p_row[a] < 1e-8:
                    continue
                next_pos = _apply_pos(ego_pos, a)
                risk_at_dest = risk_t1.get(next_pos, 0.0)
                cost = self.lambda_1 * float(lp_row[a]) + self.lambda_2 * risk_at_dest
                weights[a] = math.exp(-cost)

            final_actions.append(self._sample_action(weights, do_sample))

        return final_actions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action_probs(self, observations) -> torch.Tensor:
        """
        Get action probability distributions for all agents using
        the full context window.

        Returns:
            Tensor of shape (num_agents, 5).
        """
        inputs = []
        for i in range(self.num_agents):
            context = self._get_sorted_context_agents(i, observations)
            inputs.append(self._build_input(i, observations, self.cfg.num_agents, context))
        return self._forward_batch(inputs, self.encoder)

    def simulate_neighbor(
        self,
        neighbor_idx: int,
        ego_idx: int,
        observations,
    ) -> torch.Tensor:
        """
        Simulate a single neighbor's action from the ego's perspective.
        Uses reduced context window (single step).

        Returns:
            Tensor of shape (5,) — action probabilities.
        """
        ego_visible = set(self._get_visible_neighbors(ego_idx, observations))
        ego_visible.add(ego_idx)
        ego_visible.add(neighbor_idx)

        pool = list(ego_visible)
        context = self._get_sorted_context_agents(neighbor_idx, observations, pool)
        inp = self._build_input(
            neighbor_idx, observations, self.sim_num_agents, context
        )
        probs = self._forward_batch([inp], self.sim_encoder)
        return probs[0]

    def simulate_visible_neighbors(
        self,
        ego_idx: int,
        observations,
        only_higher_priority: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Simulate all visible neighbors of ego (single step, batched).

        Returns:
            Dict[neighbor_idx -> Tensor of shape (5,)].
        """
        visible = self._get_visible_neighbors(ego_idx, observations)
        if only_higher_priority:
            visible = [
                n for n in visible
                if self.priorities[n] < self.priorities[ego_idx]
            ]

        if not visible:
            return {}

        ego_visible_set = set(self._get_visible_neighbors(ego_idx, observations))
        ego_visible_set.add(ego_idx)

        inputs = []
        for n in visible:
            pool = list(ego_visible_set | {n})
            context = self._get_sorted_context_agents(n, observations, pool)
            inputs.append(
                self._build_input(n, observations, self.sim_num_agents, context)
            )
        probs = self._forward_batch(inputs, self.sim_encoder)
        return {n: probs[i] for i, n in enumerate(visible)}

    def get_safe_action(
        self,
        observations,
        do_sample: bool = True,
    ) -> List[int]:
        """
        Select safe actions for all agents simultaneously.

        When horizon=1: single-step hard-mask approach.
        When horizon>1: risk map-based cost minimization.
        """
        if self.horizon <= 1:
            return self._get_safe_action_single_step(observations, do_sample)
        else:
            return self._get_safe_action_multistep(observations, do_sample)

    def act(self, observations) -> List[int]:
        """
        Drop-in replacement for MAPFGPTInference.act().
        Updates state, then returns safe actions.
        """
        if self.num_agents is None:
            self._initialize(observations)
        else:
            self._update_histories(observations)
        return self.get_safe_action(observations)

    def act_with_info(self, observations) -> dict:
        """
        Like act(), but also returns diagnostics.

        Returns dict with:
            actions: List[int] — chosen safe actions (all simultaneous)
            probs: Tensor (N, 5) — raw policy probabilities
            priorities: List[int] — priority assignments
            next_positions: Dict[int, tuple] — each agent's next position
        """
        if self.num_agents is None:
            self._initialize(observations)
        else:
            self._update_histories(observations)

        all_probs = self.get_action_probs(observations)
        actions = self.get_safe_action(observations)

        positions = [tuple(obs["global_xy"]) for obs in observations]
        next_positions = {}
        for i, a in enumerate(actions):
            next_positions[i] = _apply_pos(positions[i], a)

        return {
            "actions": actions,
            "probs": all_probs,
            "priorities": self.priorities,
            "next_positions": next_positions,
        }


# ---------------------------------------------------------------------------
# Toolbox-compatible config and adapter for benchmark registration
# ---------------------------------------------------------------------------

from pogema_toolbox.algorithm_config import AlgoBase
from pydantic import Extra


class CollisionCountingMAPFGPTConfig(AlgoBase, extra=Extra.forbid):
    """
    Config for the collision-counting baseline wrapper.
    Mirrors MAPFGPTInferenceConfig but registers under "MAPF-GPT-Counted"
    so the evaluator can track vertex/edge collisions for the plain baseline.
    """

    name: Literal["MAPF-GPT-Counted"] = "MAPF-GPT-Counted"

    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    path_to_weights: Optional[str] = "weights/model-6M.pt"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    repo_id: str = "aandreychuk/MAPF-GPT"


class CollisionCountingMAPFGPT:
    """
    Thin adapter that runs plain MAPFGPTInference while counting vertex/edge
    collisions each step.  Registered under "MAPF-GPT-Counted" so benchmark
    scripts can compare baseline and safe variants on equal footing.

    Call get_extra_metrics() after an episode to retrieve per-episode totals.
    """

    def __init__(self, cfg: CollisionCountingMAPFGPTConfig):
        mapf_cfg = MAPFGPTInferenceConfig(
            name="MAPF-GPT",
            num_agents=cfg.num_agents,
            num_previous_actions=cfg.num_previous_actions,
            cost2go_value_limit=cfg.cost2go_value_limit,
            agents_radius=cfg.agents_radius,
            cost2go_radius=cfg.cost2go_radius,
            path_to_weights=cfg.path_to_weights,
            device=cfg.device,
            context_size=cfg.context_size,
            mask_actions_history=cfg.mask_actions_history,
            mask_goal=cfg.mask_goal,
            mask_cost2go=cfg.mask_cost2go,
            mask_greed_action=cfg.mask_greed_action,
            repo_id=cfg.repo_id,
        )
        self._algo = MAPFGPTInference(mapf_cfg)
        self._collision_vertex: int = 0
        self._collision_edge: int = 0

    def act(self, observations):
        positions = [tuple(o["global_xy"]) for o in observations]
        actions = self._algo.act(observations)
        stats = count_collisions(positions, actions)
        self._collision_vertex += stats["vertex"]
        self._collision_edge += stats["edge"]
        return actions

    def reset_states(self):
        self._collision_vertex = 0
        self._collision_edge = 0
        return self._algo.reset_states()

    def get_extra_metrics(self) -> dict:
        """Return per-episode collision totals to be merged into benchmark metrics."""
        return {
            "collision_vertex": self._collision_vertex,
            "collision_edge": self._collision_edge,
            "collision_total": self._collision_vertex + self._collision_edge,
        }


class DecentralizedWrapperConfig(AlgoBase, extra=Extra.forbid):
    """
    Pydantic config for DecentralizedWrapper that can be registered with
    ToolboxRegistry and used in benchmark YAML files.

    Combines all MAPFGPTInferenceConfig fields (model weights, device, etc.)
    with the wrapper-specific hyperparameters so a single config object
    describes the full algorithm.
    """

    name: Literal["MAPF-GPT-Safe"] = "MAPF-GPT-Safe"

    # ---- model fields (mirrors MAPFGPTInferenceConfig) ----
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    path_to_weights: Optional[str] = "weights/model-2M.pt"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    repo_id: str = "aandreychuk/MAPF-GPT"

    # ---- wrapper-specific fields ----
    priority_scheme: Literal["index", "random"] = "index"
    sim_num_agents: int = 1
    horizon: int = 3
    epsilon: float = 0.1
    alpha: float = 0.5
    lambda_1: float = 1.0
    lambda_2: float = 1.0
    sequential_simulation: bool = False
    conflict_radius: Optional[int] = None  # None = use agents_radius


class DecentralizedWrapperAlgo:
    """
    Thin adapter that wraps DecentralizedWrapper with the act() / reset_states()
    interface expected by the pogema-toolbox evaluator.

    Registered under the name "MAPF-GPT-Safe" so benchmark YAML files can
    refer to it alongside "MAPF-GPT".

    Also tracks vertex/edge collisions each step via count_collisions().
    Call get_extra_metrics() after an episode to retrieve per-episode totals.
    """

    def __init__(self, cfg: DecentralizedWrapperConfig):
        mapf_cfg = MAPFGPTInferenceConfig(
            name="MAPF-GPT",
            num_agents=cfg.num_agents,
            num_previous_actions=cfg.num_previous_actions,
            cost2go_value_limit=cfg.cost2go_value_limit,
            agents_radius=cfg.agents_radius,
            cost2go_radius=cfg.cost2go_radius,
            path_to_weights=cfg.path_to_weights,
            device=cfg.device,
            context_size=cfg.context_size,
            mask_actions_history=cfg.mask_actions_history,
            mask_goal=cfg.mask_goal,
            mask_cost2go=cfg.mask_cost2go,
            mask_greed_action=cfg.mask_greed_action,
            repo_id=cfg.repo_id,
        )
        self._wrapper = DecentralizedWrapper(
            mapf_cfg,
            priority_scheme=cfg.priority_scheme,
            sim_num_agents=cfg.sim_num_agents,
            horizon=cfg.horizon,
            epsilon=cfg.epsilon,
            alpha=cfg.alpha,
            lambda_1=cfg.lambda_1,
            lambda_2=cfg.lambda_2,
            sequential_simulation=cfg.sequential_simulation,
            conflict_radius=cfg.conflict_radius,
        )
        self._collision_vertex: int = 0
        self._collision_edge: int = 0

    def act(self, observations):
        positions = [tuple(o["global_xy"]) for o in observations]
        actions = self._wrapper.act(observations)
        stats = count_collisions(positions, actions)
        self._collision_vertex += stats["vertex"]
        self._collision_edge += stats["edge"]
        return actions

    def reset_states(self):
        self._collision_vertex = 0
        self._collision_edge = 0
        return self._wrapper.reset_states()

    def get_extra_metrics(self) -> dict:
        """Return per-episode collision totals to be merged into benchmark metrics."""
        return {
            "collision_vertex": self._collision_vertex,
            "collision_edge": self._collision_edge,
            "collision_total": self._collision_vertex + self._collision_edge,
        }
