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

from typing import Dict, List, Literal, Optional

import cppimport.import_hook
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from pathlib import Path

from gpt.model import GPT, GPTConfig
from gpt.inference import MAPFGPTInferenceConfig, strip_prefix_from_state_dict
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

    def _collect_hp_neighbors(self, observations) -> set:
        """Collect all unique higher-priority neighbors across all egos."""
        hp_neighbors = set()
        for ego_idx in range(self.num_agents):
            visible = self._get_visible_neighbors(ego_idx, observations)
            for n in visible:
                if self.priorities[n] < self.priorities[ego_idx]:
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

        Propagates probability mass through all actions above epsilon.
        Uses simplified history: each frontier position keeps the history
        from its highest-mass predecessor.

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
            all_probs = self._forward_batch(inputs, self.sim_encoder)

            # Propagate mass through surviving actions
            new_frontier: Dict[tuple, tuple] = {}
            step_occ: Dict[tuple, float] = {}

            for idx, (pos, (mass, hist)) in enumerate(frontier_items):
                probs = all_probs[idx]

                # In sequential mode, penalize actions landing on
                # higher-priority agents' occupancy
                if prior_occupancy is not None and t < len(prior_occupancy):
                    adjusted = probs.clone()
                    for a in range(5):
                        next_p = _apply_pos(pos, a)
                        occ_risk = prior_occupancy[t].get(next_p, 0.0)
                        if occ_risk > 0:
                            adjusted[a] = adjusted[a] * (1.0 - occ_risk)
                    # Renormalize after penalty
                    adj_total = adjusted.sum()
                    if adj_total > 0:
                        probs = adjusted / adj_total

                pruned = self._prune_probs(probs)

                for a in range(5):
                    p = pruned[a].item()
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

        In independent mode: all neighbors simulated independently.
        In sequential mode: neighbors simulated in priority order,
        each accounting for higher-priority agents' occupancy.

        Returns:
            Dict[neighbor_idx -> list of h occupancy dicts].
            Each occupancy dict maps (x, y) -> probability mass.
        """
        all_hp_neighbors = self._collect_hp_neighbors(observations)
        if not all_hp_neighbors:
            return {}

        # Sort by priority (highest priority = lowest value first)
        hp_list = sorted(all_hp_neighbors, key=lambda n: self.priorities[n])

        trajectories: Dict[int, List[Dict[tuple, float]]] = {}

        if not self.sequential_simulation:
            # Independent mode: batch all agents' frontiers together per step
            # For simplicity, simulate each agent's tree separately but
            # batch the forward passes across all agents at each step
            for n in hp_list:
                pos = tuple(observations[n]["global_xy"])
                target = tuple(observations[n]["global_target_xy"])
                history = list(self.action_histories[n])
                trajectories[n] = self._simulate_single_agent_tree(
                    pos, target, history, horizon
                )
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

        Only propagates toward the goal (decreasing distance), modeling
        the likely future path of a higher-priority agent.
        """
        d = self.cost2go_data[target]
        propagated = dict(risk_map)

        # Process cells in decreasing distance order (far from goal first)
        # so propagated risk flows toward the goal
        cells = []
        for (i, j), risk in risk_map.items():
            dist = d[i][j]
            if dist >= 0 and risk > 1e-8:
                cells.append(((i, j), risk, dist))
        cells.sort(key=lambda x: -x[2])

        for (i, j), risk, d_ij in cells:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x, y = i + dx, j + dy
                try:
                    d_xy = d[x][y]
                except IndexError:
                    continue
                if d_xy < 0 or d_xy >= d_ij:
                    continue
                propagated_risk = risk * (self.alpha ** (d_ij - d_xy))
                if propagated_risk > 1e-8:
                    propagated[(x, y)] = max(
                        propagated.get((x, y), 0.0), propagated_risk
                    )

        return propagated

    def _build_cumulative_risk_map(
        self, ego_idx: int, observations, trajectories: dict
    ) -> Dict[tuple, float]:
        """
        Build per-ego cumulative risk map from higher-priority neighbors (Eq. 3).

        For each hp neighbor visible to ego:
        1. Propagate that neighbor's occupancy along its goal direction.
        2. Aggregate across neighbors using element-wise max.

        Returns:
            Dict[(x,y) -> risk]. Covers all timesteps, already propagated.
        """
        visible = self._get_visible_neighbors(ego_idx, observations)
        hp_neighbors = [
            n for n in visible
            if self.priorities[n] < self.priorities[ego_idx] and n in trajectories
        ]

        if not hp_neighbors:
            return {}

        cumulative_risk: Dict[tuple, float] = {}

        for n in hp_neighbors:
            n_target = tuple(observations[n]["global_target_xy"])
            # Merge all timesteps for this neighbor into a single risk map
            # (take max across timesteps for each cell)
            neighbor_risk: Dict[tuple, float] = {}
            for step_occ in trajectories[n]:
                for cell, prob in step_occ.items():
                    neighbor_risk[cell] = max(
                        neighbor_risk.get(cell, 0.0), prob
                    )
            # Propagate along this neighbor's goal direction
            propagated = self._propagate_risk(neighbor_risk, n_target)
            # Aggregate with max across neighbors (Eq. 3)
            for cell, risk in propagated.items():
                cumulative_risk[cell] = max(
                    cumulative_risk.get(cell, 0.0), risk
                )

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
        final_actions = []
        for ego_idx in range(self.num_agents):
            ego_pos = positions[ego_idx]
            probs = all_probs[ego_idx]
            risk = ego_risk_maps[ego_idx]

            best_action = 0
            best_cost = float('inf')
            for a in range(5):
                p = probs[a].item()
                if p < 1e-8:
                    continue
                next_pos = _apply_pos(ego_pos, a)
                log_prob = -torch.log(probs[a]).item()
                risk_at_dest = risk.get(next_pos, 0.0)
                cost = self.lambda_1 * log_prob + self.lambda_2 * risk_at_dest
                if cost < best_cost:
                    best_cost = cost
                    best_action = a

            final_actions.append(best_action)

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
