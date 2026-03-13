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


class DecentralizedWrapper:
    """
    Fully decentralized safety wrapper with fixed priorities and
    multi-step occupancy-based safety cost field.

    Each agent independently:
    1. Runs the shared policy to get its own action distribution.
    2. Simulates higher-priority visible neighbors' trajectories for
       multiple steps, building a probabilistic occupancy field.
    3. Simulates its own future trajectory for each candidate action.
    4. Penalizes actions whose trajectories overlap with the occupancy
       field, then samples from the adjusted distribution.

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
        gamma: float = 0.9,
        safety_lambda: float = 5.0,
    ):
        """
        Args:
            cfg: Standard MAPF-GPT inference config.
            priority_scheme: How to assign priorities.
                "index" = agent 0 has highest priority.
                "random" = random shuffle at episode start.
            sim_num_agents: Number of agent slots in reduced context window
                for neighbor/ego forward simulation.
            horizon: Number of steps to simulate forward. 1 = single-step
                hard-mask (original behavior). >1 = multi-step cost field.
            gamma: Discount factor for future occupancy danger (0 < gamma <= 1).
            safety_lambda: Penalty strength. Higher = more conservative.
                P_adjusted(a) ∝ P_policy(a) * exp(-lambda * danger(a))
        """
        self.cfg = cfg
        self.priority_scheme = priority_scheme
        self.sim_num_agents = sim_num_agents
        self.horizon = horizon
        self.gamma = gamma
        self.safety_lambda = safety_lambda

        # Load model
        path_to_weights = Path(cfg.path_to_weights)
        if path_to_weights.name in ['model-2M.pt', 'model-6M.pt', 'model-85M.pt']:
            hf_hub_download(repo_id=cfg.repo_id, filename=path_to_weights.name,
                            local_dir=path_to_weights.parent)

        if cfg.device == 'cuda' and not torch.cuda.is_available():
            cfg.device = 'cpu'
        elif cfg.device == 'mps' and not torch.backends.mps.is_available():
            cfg.device = 'cpu'

        checkpoint = torch.load(path_to_weights, map_location=cfg.device)
        model_state_dict = strip_prefix_from_state_dict(checkpoint["model"])
        gpt_config = GPTConfig(**checkpoint.get("model_args"))
        self.net = GPT(gpt_config)
        self.net.load_state_dict(model_state_dict, strict=False)
        self.net.to(cfg.device)
        self.net.eval()

        # Encoder for full observations (ego agent's own action)
        self.encoder = Encoder(InputParameters(
            num_agents=cfg.num_agents,
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

        # Encoder for reduced-window simulation (neighbors + ego rollout)
        self.sim_encoder = Encoder(InputParameters(
            num_agents=sim_num_agents,
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
        has been hypothetically advanced. Uses sim_num_agents=1 context
        (only the agent itself).
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
    # Multi-step simulation
    # ------------------------------------------------------------------

    def _simulate_hp_trajectories(
        self, observations, horizon: int
    ) -> Dict[int, List[Dict[tuple, float]]]:
        """
        Simulate all unique higher-priority neighbors' trajectories.

        At each step:
        - Run the shared policy to get the FULL probability distribution
          over 5 actions.
        - Record the distribution as occupancy: each reachable cell gets
          the probability mass of the action leading there.
        - Advance the position by argmax for the next step's simulation.

        This gives a soft occupancy field — cells along the most likely
        path get high probability, but adjacent cells also get nonzero
        mass from alternative actions.

        Returns:
            Dict[neighbor_idx -> list of h occupancy dicts].
            Each occupancy dict maps (x, y) -> probability.
        """
        # Collect all unique neighbors that any ego needs simulated
        all_hp_neighbors = set()
        for ego_idx in range(self.num_agents):
            visible = self._get_visible_neighbors(ego_idx, observations)
            for n in visible:
                if self.priorities[n] < self.priorities[ego_idx]:
                    all_hp_neighbors.add(n)

        if not all_hp_neighbors:
            return {}

        hp_list = sorted(all_hp_neighbors)

        # Initialize simulated state for each neighbor
        sim_states = {}
        for n in hp_list:
            sim_states[n] = {
                "pos": tuple(observations[n]["global_xy"]),
                "target": tuple(observations[n]["global_target_xy"]),
                "history": list(self.action_histories[n]),
            }

        trajectories: Dict[int, List[Dict[tuple, float]]] = {n: [] for n in hp_list}

        for t in range(horizon):
            # Build inputs for all neighbors at current simulated positions
            inputs = []
            for n in hp_list:
                s = sim_states[n]
                inputs.append(
                    self._build_sim_input(s["pos"], s["target"], s["history"])
                )

            # One batched forward pass for all neighbors at this step
            all_probs = self._forward_batch(inputs, self.sim_encoder)

            for i, n in enumerate(hp_list):
                probs = all_probs[i]
                s = sim_states[n]

                # Record full probability distribution as occupancy
                step_occ: Dict[tuple, float] = {}
                for a in range(5):
                    dx, dy = ACTION_TO_DELTA[a]
                    next_pos = (s["pos"][0] + dx, s["pos"][1] + dy)
                    p = probs[a].item()
                    if p > 1e-6:
                        step_occ[next_pos] = step_occ.get(next_pos, 0.0) + p
                trajectories[n].append(step_occ)

                # Advance by argmax for next step
                best = torch.argmax(probs).item()
                dx, dy = ACTION_TO_DELTA[best]
                s["pos"] = (s["pos"][0] + dx, s["pos"][1] + dy)
                s["history"] = list(s["history"]) + [MOVES_STR[best]]
                s["history"] = s["history"][-self.cfg.num_previous_actions:]

        return trajectories

    def _build_ego_occupancy(
        self, ego_idx: int, observations, trajectories: dict
    ) -> List[Dict[tuple, float]]:
        """
        Build per-ego occupancy field from higher-priority neighbors.

        Only includes neighbors that are visible to the ego agent.
        Sums occupancy probabilities from all relevant neighbors at
        each cell and step.

        Returns:
            List of h occupancy dicts. occupancy[t][(x,y)] = total
            probability of any higher-priority neighbor being there.
        """
        visible = self._get_visible_neighbors(ego_idx, observations)
        hp_neighbors = [
            n for n in visible
            if self.priorities[n] < self.priorities[ego_idx] and n in trajectories
        ]

        if not hp_neighbors:
            return [{} for _ in range(len(next(iter(trajectories.values())))) ] if trajectories else []

        horizon = len(trajectories[hp_neighbors[0]])
        occupancy = []
        for t in range(horizon):
            step_occ: Dict[tuple, float] = {}
            for n in hp_neighbors:
                for pos, prob in trajectories[n][t].items():
                    step_occ[pos] = step_occ.get(pos, 0.0) + prob
            occupancy.append(step_occ)
        return occupancy

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

        When horizon=1: single-step hard-mask approach. Each agent
        masks actions that conflict with higher-priority neighbors'
        predicted argmax moves.

        When horizon>1: multi-step cost field approach.
        1. Simulate higher-priority neighbors for `horizon` steps,
           recording the full probability distribution at each step
           as a soft occupancy field.
        2. For each ego candidate action, simulate ego forward for
           `horizon` steps and accumulate danger from the occupancy
           field with gamma discount.
        3. Adjust action probabilities:
           P_adjusted(a) ∝ P_policy(a) * exp(-lambda * danger(a))

        Args:
            observations: Per-agent observations from the environment.
            do_sample: True = sample, False = argmax.

        Returns:
            List of action indices, one per agent.
        """
        if self.horizon <= 1:
            return self._get_safe_action_single_step(observations, do_sample)
        else:
            return self._get_safe_action_multistep(observations, do_sample)

    def _get_safe_action_single_step(
        self, observations, do_sample: bool
    ) -> List[int]:
        """Original single-step hard-mask approach."""
        num_agents = self.num_agents
        positions = [tuple(obs["global_xy"]) for obs in observations]

        all_probs = self.get_action_probs(observations)

        # Batch all neighbor simulations
        sim_requests = []
        sim_inputs = []
        for ego_idx in range(num_agents):
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

        hp_predicted_next: Dict[int, Dict[int, tuple]] = {}
        for i, (ego_idx, n) in enumerate(sim_requests):
            predicted_action = torch.argmax(sim_probs[i]).item()
            dx, dy = ACTION_TO_DELTA[predicted_action]
            n_next = (positions[n][0] + dx, positions[n][1] + dy)
            if ego_idx not in hp_predicted_next:
                hp_predicted_next[ego_idx] = {}
            hp_predicted_next[ego_idx][n] = n_next

        final_actions = [0] * num_agents
        for ego_idx in range(num_agents):
            ego_pos = positions[ego_idx]
            ego_probs = all_probs[ego_idx].clone()
            neighbors_next = hp_predicted_next.get(ego_idx, {})

            for action_idx in range(5):
                dx, dy = ACTION_TO_DELTA[action_idx]
                ego_next = (ego_pos[0] + dx, ego_pos[1] + dy)
                for other_idx, other_next in neighbors_next.items():
                    other_pos = positions[other_idx]
                    if ego_next == other_next:
                        ego_probs[action_idx] = 0.0
                        break
                    if ego_next == other_pos and other_next == ego_pos:
                        ego_probs[action_idx] = 0.0
                        break

            total = ego_probs.sum()
            if total == 0:
                action = 0
            elif do_sample:
                ego_probs = ego_probs / total
                action = torch.multinomial(ego_probs, num_samples=1).item()
            else:
                action = torch.argmax(ego_probs).item()
            final_actions[ego_idx] = action

        return final_actions

    def _get_safe_action_multistep(
        self, observations, do_sample: bool
    ) -> List[int]:
        """
        Multi-step cost field approach.

        Forward passes (all batched):
        - 1 pass: ego action probs (N agents, full context)
        - h passes: neighbor trajectory simulation (unique hp neighbors)
        - h-1 passes: ego trajectory rollout (N*5 candidates)
        Total: 2h forward passes.
        """
        num_agents = self.num_agents
        h = self.horizon
        positions = [tuple(obs["global_xy"]) for obs in observations]
        targets = [tuple(obs["global_target_xy"]) for obs in observations]

        # --- 1. Ego action probs (full context) ---
        all_probs = self.get_action_probs(observations)

        # --- 2. Simulate higher-priority neighbors' trajectories ---
        trajectories = self._simulate_hp_trajectories(observations, h)

        # --- 3. Build per-ego occupancy fields ---
        ego_occupancies = []
        for ego_idx in range(num_agents):
            if trajectories:
                occ = self._build_ego_occupancy(ego_idx, observations, trajectories)
            else:
                occ = [{} for _ in range(h)]
            ego_occupancies.append(occ)

        # --- 4. Simulate ego trajectory for each candidate action ---
        # Initialize: for each ego × each candidate action, compute
        # position after taking that action and updated history.
        # sim_entries[ego_idx * 5 + action] = state
        sim_entries = []
        danger = torch.zeros(num_agents, 5, device=self.cfg.device)

        for ego_idx in range(num_agents):
            ego_pos = positions[ego_idx]
            ego_history = self.action_histories[ego_idx]

            for a in range(5):
                dx, dy = ACTION_TO_DELTA[a]
                next_pos = (ego_pos[0] + dx, ego_pos[1] + dy)
                new_history = list(ego_history) + [MOVES_STR[a]]
                new_history = new_history[-self.cfg.num_previous_actions:]

                sim_entries.append({
                    "pos": next_pos,
                    "target": targets[ego_idx],
                    "history": new_history,
                })

                # Danger at t=0: occupancy at the position ego moves to
                occ = ego_occupancies[ego_idx]
                if occ:
                    danger[ego_idx, a] += occ[0].get(next_pos, 0.0)

                    # Also check edge conflict at t=0: ego goes to
                    # neighbor's current pos while neighbor goes to
                    # ego's current pos
                    visible = self._get_visible_neighbors(ego_idx, observations)
                    for n in visible:
                        if self.priorities[n] >= self.priorities[ego_idx]:
                            continue
                        if n not in trajectories:
                            continue
                        n_pos = positions[n]
                        if next_pos == n_pos:
                            # Check if neighbor is moving to ego's pos
                            n_to_ego_prob = trajectories[n][0].get(ego_pos, 0.0)
                            danger[ego_idx, a] += n_to_ego_prob

        # Steps t=1..h-1: roll out ego trajectories in batches
        for t in range(1, h):
            # Build sim inputs for all N*5 entries
            inputs = []
            for entry in sim_entries:
                inputs.append(
                    self._build_sim_input(
                        entry["pos"], entry["target"], entry["history"]
                    )
                )

            # One batched forward pass for all ego candidates
            step_probs = self._forward_batch(inputs, self.sim_encoder)

            # Advance each entry by argmax, accumulate danger
            for idx, entry in enumerate(sim_entries):
                ego_idx = idx // 5
                a = idx % 5

                best = torch.argmax(step_probs[idx]).item()
                dx, dy = ACTION_TO_DELTA[best]
                entry["pos"] = (entry["pos"][0] + dx, entry["pos"][1] + dy)
                entry["history"] = (
                    list(entry["history"]) + [MOVES_STR[best]]
                )[-self.cfg.num_previous_actions:]

                # Accumulate discounted danger
                occ = ego_occupancies[ego_idx]
                if t < len(occ):
                    danger[ego_idx, a] += (
                        (self.gamma ** t) * occ[t].get(entry["pos"], 0.0)
                    )

        # --- 5. Adjust probabilities and select actions ---
        final_actions = [0] * num_agents
        adjusted = all_probs * torch.exp(-self.safety_lambda * danger)

        for ego_idx in range(num_agents):
            ego_adjusted = adjusted[ego_idx]
            total = ego_adjusted.sum()
            if total == 0:
                final_actions[ego_idx] = 0  # wait as fallback
            elif do_sample:
                ego_adjusted = ego_adjusted / total
                final_actions[ego_idx] = torch.multinomial(
                    ego_adjusted, num_samples=1
                ).item()
            else:
                final_actions[ego_idx] = torch.argmax(ego_adjusted).item()

        return final_actions

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
            dx, dy = ACTION_TO_DELTA[a]
            next_positions[i] = (positions[i][0] + dx, positions[i][1] + dy)

        return {
            "actions": actions,
            "probs": all_probs,
            "priorities": self.priorities,
            "next_positions": next_positions,
        }
