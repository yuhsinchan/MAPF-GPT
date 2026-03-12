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


class DecentralizedWrapper:
    """
    Fully decentralized safety wrapper with fixed priorities.

    Each agent independently:
    1. Runs the shared policy to get its own action distribution.
    2. Simulates higher-priority visible neighbors' actions using the
       same policy with a reduced context window.
    3. Selects a safe action that avoids vertex/edge conflicts with
       predicted higher-priority moves.

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
    ):
        """
        Args:
            cfg: Standard MAPF-GPT inference config.
            priority_scheme: How to assign priorities.
                "index" = agent 0 has highest priority.
                "random" = random shuffle at episode start.
            sim_num_agents: Number of agent slots to use when simulating
                a neighbor's observation. Lower = faster but less context.
                Set to 1 to only include the simulated agent itself.
        """
        self.cfg = cfg
        self.priority_scheme = priority_scheme
        self.sim_num_agents = sim_num_agents

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

        # Encoder for full observations (ego agent)
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

        # Encoder for reduced-window neighbor simulation
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
        self.priority_order: Optional[List[int]] = None
        self.cost2go_data = None
        self.action_histories: Optional[List[List[str]]] = None
        self.position_histories: Optional[List[List]] = None

    def reset_states(self):
        self.num_agents = None
        self.priorities = None
        self.priority_order = None
        self.cost2go_data = None
        self.action_histories = None
        self.position_histories = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _initialize(self, observations):
        n = len(observations)
        self.num_agents = n

        # Assign priorities (lower value = higher priority)
        if self.priority_scheme == "random":
            import random
            self.priorities = list(range(n))
            random.shuffle(self.priorities)
        else:
            self.priorities = list(range(n))

        # priority_order[0] = agent with highest priority (acts first)
        self.priority_order = sorted(range(n), key=lambda i: self.priorities[i])

        # Cost2go is precomputed from the static obstacle map
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
        """
        Get agents sorted by cost2go distance from ego, as generate_input does.

        Args:
            candidate_pool: If given, only consider these agent indices.
                Otherwise consider all agents.
        """
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
        """
        Build tokenizer input dict for a single agent.

        Args:
            ego_idx: The agent to build the observation for.
            max_agents: How many agent slots to fill (self.cfg.num_agents
                for ego, self.sim_num_agents for simulation).
            context_agents: Sorted list of agent indices to include.
        """
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

    # ------------------------------------------------------------------
    # Forward pass helpers
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
        Simulate a single neighbor's action from the ego agent's perspective.

        Constructs the neighbor's observation using only agents visible to
        the ego agent (intersection of views), with a reduced context window.

        Returns:
            Tensor of shape (5,) — action probabilities for the neighbor.
        """
        # Only include agents that ego can see (decentralized constraint)
        ego_visible = set(self._get_visible_neighbors(ego_idx, observations))
        ego_visible.add(ego_idx)  # ego is visible to the neighbor
        ego_visible.add(neighbor_idx)  # neighbor sees itself

        # Sort by distance from the neighbor's perspective
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
        Simulate all visible neighbors of ego (optionally only higher priority).

        Returns:
            Dict[neighbor_idx -> Tensor of shape (5,)] with action probs.
        """
        visible = self._get_visible_neighbors(ego_idx, observations)
        if only_higher_priority:
            visible = [
                n for n in visible
                if self.priorities[n] < self.priorities[ego_idx]
            ]

        if not visible:
            return {}

        # Ego's visible set (used to restrict neighbor context)
        ego_visible_set = set(self._get_visible_neighbors(ego_idx, observations))
        ego_visible_set.add(ego_idx)

        # Batch all neighbor simulations
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
        conflict_penalty: float = 0.0,
        do_sample: bool = True,
    ) -> List[int]:
        """
        Decentralized safe action selection with fixed priorities.

        For each agent (processed in priority order):
        1. Get ego action probabilities (full context window).
        2. Simulate higher-priority visible neighbors using the shared
           policy with reduced context window.
        3. Treat higher-priority agents' predicted actions as committed.
        4. Mask/penalize ego actions that conflict, sample from remainder.

        Since all agents use the same deterministic policy and priorities,
        they independently arrive at consistent predictions without
        any communication.

        Args:
            observations: Per-agent observations from the environment.
            conflict_penalty: 0.0 = hard mask, >0 = soft penalty.
            do_sample: True = sample, False = argmax.

        Returns:
            List of action indices, one per agent.
        """
        num_agents = self.num_agents
        positions = [tuple(obs["global_xy"]) for obs in observations]

        # 1. Get all agents' action probs in one batch (full context)
        all_probs = self.get_action_probs(observations)

        # 2. Committed actions: processed in priority order
        #    committed_next[agent_idx] = predicted next position
        committed_next: Dict[int, tuple] = {}
        final_actions = [0] * num_agents

        for agent_idx in self.priority_order:
            ego_pos = positions[agent_idx]
            ego_probs = all_probs[agent_idx].clone()

            # 3. Simulate higher-priority visible neighbors
            visible = self._get_visible_neighbors(agent_idx, observations)
            higher_priority_visible = [
                n for n in visible
                if self.priorities[n] < self.priorities[agent_idx]
            ]

            # For higher-priority agents, use their committed actions
            # (they've already been processed)
            hp_next_positions = {}
            for n in higher_priority_visible:
                if n in committed_next:
                    hp_next_positions[n] = committed_next[n]
                else:
                    # Shouldn't happen if processed in priority order,
                    # but fall back to simulated argmax
                    sim_probs = self.simulate_neighbor(n, agent_idx, observations)
                    predicted_action = torch.argmax(sim_probs).item()
                    dx, dy = ACTION_TO_DELTA[predicted_action]
                    hp_next_positions[n] = (
                        positions[n][0] + dx, positions[n][1] + dy
                    )

            # 4. Check each candidate action for conflicts with
            #    higher-priority committed moves
            for action_idx in range(5):
                dx, dy = ACTION_TO_DELTA[action_idx]
                ego_next = (ego_pos[0] + dx, ego_pos[1] + dy)

                for other_idx, other_next in hp_next_positions.items():
                    other_pos = positions[other_idx]

                    # Vertex conflict
                    if ego_next == other_next:
                        if conflict_penalty == 0.0:
                            ego_probs[action_idx] = 0.0
                        else:
                            ego_probs[action_idx] *= conflict_penalty
                        break

                    # Edge conflict (swap)
                    if ego_next == other_pos and other_next == ego_pos:
                        if conflict_penalty == 0.0:
                            ego_probs[action_idx] = 0.0
                        else:
                            ego_probs[action_idx] *= conflict_penalty
                        break

            # 5. Select action from adjusted distribution
            total = ego_probs.sum()
            if total == 0:
                action = 0  # wait as safe fallback
            elif do_sample:
                ego_probs = ego_probs / total
                action = torch.multinomial(ego_probs, num_samples=1).item()
            else:
                action = torch.argmax(ego_probs).item()

            final_actions[agent_idx] = action

            # Commit this agent's action
            dx, dy = ACTION_TO_DELTA[action]
            committed_next[agent_idx] = (ego_pos[0] + dx, ego_pos[1] + dy)

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
            actions: List[int]
            probs: Tensor (N, 5) — raw policy probabilities
            priorities: List[int] — priority assignments
            committed_positions: Dict[int, tuple] — each agent's committed
                next position in priority order
        """
        if self.num_agents is None:
            self._initialize(observations)
        else:
            self._update_histories(observations)

        all_probs = self.get_action_probs(observations)
        actions = self.get_safe_action(observations)

        committed = {}
        positions = [tuple(obs["global_xy"]) for obs in observations]
        for i, a in enumerate(actions):
            dx, dy = ACTION_TO_DELTA[a]
            committed[i] = (positions[i][0] + dx, positions[i][1] + dy)

        return {
            "actions": actions,
            "probs": all_probs,
            "priorities": self.priorities,
            "committed_positions": committed,
        }
