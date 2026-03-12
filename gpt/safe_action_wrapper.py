"""
SafeActionWrapper for MAPF-GPT agents.

Wraps MAPFGPTInference to provide:
1. Action probability distributions from the policy network.
2. Simulation of neighbor agents' next steps using the shared policy.
3. Safe action selection that avoids vertex and edge conflicts with
   simulated neighbor trajectories.

The core idea: since all agents share the same policy network, each agent
can forward-simulate what its neighbors will do without communication.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

# Action index <-> displacement mapping (matches POGEMA conventions)
ACTION_TO_DELTA = {
    0: (0, 0),   # wait
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}


class SafeActionWrapper:
    """
    Wraps MAPFGPTInference to add safety-aware action selection.

    Usage (drop-in replacement for MAPFGPTInference):
        wrapper = SafeActionWrapper(cfg)
        wrapper.reset_states()
        actions = wrapper.act(observations)  # returns safe actions

    Lower-level API:
        probs = wrapper.get_all_action_probs(observations)  # (N, 5) tensor
        neighbor_probs = wrapper.simulate_neighbors(ego_idx, observations)
    """

    def __init__(self, cfg: MAPFGPTInferenceConfig):
        self.inference = MAPFGPTInference(cfg)
        self.cfg = cfg
        self._state_updated = False

    def reset_states(self):
        self.inference.reset_states()
        self._state_updated = False

    # ------------------------------------------------------------------
    # Core: single batched forward pass returning probs for all agents
    # ------------------------------------------------------------------

    def _forward_all(self, observations) -> torch.Tensor:
        """
        Run the policy for all agents in one batched forward pass.

        Returns:
            probs: Tensor of shape (num_agents, 5) — action probabilities.
        """
        inputs = self.inference.generate_input(observations)
        tensor_obs = torch.tensor(
            [self.inference.encoder.encode(inp) for inp in inputs],
            dtype=torch.long,
            device=self.cfg.device,
        )
        with torch.no_grad():
            logits, _ = self.inference.net(tensor_obs)
            logits = logits[:, -1, :]
            # Mask to valid 5 actions
            masked = torch.full_like(logits, float("-inf"))
            masked[:, :5] = logits[:, :5]
            probs = F.softmax(masked, dim=-1)[:, :5]
        return probs

    # ------------------------------------------------------------------
    # State management (mirrors MAPFGPTInference.act)
    # ------------------------------------------------------------------

    def _update_state(self, observations):
        """Update cost2go, action history, and position history."""
        moves = {(0, 0): "w", (-1, 0): "u", (1, 0): "d", (0, -1): "l", (0, 1): "r"}
        inf = self.inference
        num_agents = len(observations)

        if inf.cost2go_data is None:
            from tokenizer import cost2go

            global_obs = observations[0]["global_obstacles"].copy().astype(int).tolist()
            inf.cost2go_data = cost2go.precompute_cost2go(
                global_obs, self.cfg.cost2go_radius
            )
            inf.actions_history = [
                ["n"] * self.cfg.num_previous_actions for _ in range(num_agents)
            ]
            inf.position_history = [[obs["global_xy"]] for obs in observations]
        else:
            for i in range(num_agents):
                inf.position_history[i].append(observations[i]["global_xy"])
                dx = inf.position_history[i][-1][0] - inf.position_history[i][-2][0]
                dy = inf.position_history[i][-1][1] - inf.position_history[i][-2][1]
                inf.actions_history[i].append(moves[(dx, dy)])
                inf.actions_history[i] = inf.actions_history[i][
                    -self.cfg.num_previous_actions :
                ]
        self._state_updated = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_action_probs(self, observations) -> torch.Tensor:
        """
        Get action probability distributions for ALL agents.

        Must be called after _update_state (or via act()).

        Returns:
            Tensor of shape (num_agents, 5).
        """
        return self._forward_all(observations)

    def get_action_logits(self, observations) -> torch.Tensor:
        """
        Get raw masked logits (before softmax) for all agents.

        Returns:
            Tensor of shape (num_agents, 5).
        """
        inputs = self.inference.generate_input(observations)
        tensor_obs = torch.tensor(
            [self.inference.encoder.encode(inp) for inp in inputs],
            dtype=torch.long,
            device=self.cfg.device,
        )
        with torch.no_grad():
            logits, _ = self.inference.net(tensor_obs)
            logits = logits[:, -1, :5]
        return logits

    def simulate_neighbors(
        self,
        ego_idx: int,
        observations,
        neighbor_indices: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Simulate what visible neighbors would do using the shared policy.

        Because every agent runs the same network, agent `ego_idx` can
        predict its neighbors' actions by running a forward pass from
        each neighbor's perspective — no communication needed.

        Args:
            ego_idx: The ego agent's index.
            observations: Full observation list (all agents).
            neighbor_indices: Specific neighbors to simulate.
                Defaults to all agents within `agents_radius` of ego.

        Returns:
            Dict[neighbor_idx -> Tensor of shape (5,)] with action probs.
        """
        if neighbor_indices is None:
            neighbor_indices = self._get_visible_neighbors(ego_idx, observations)

        if not neighbor_indices:
            return {}

        # Build inputs only for the requested neighbors
        all_inputs = self.inference.generate_input(observations)
        selected_inputs = [all_inputs[n] for n in neighbor_indices]

        tensor_obs = torch.tensor(
            [self.inference.encoder.encode(inp) for inp in selected_inputs],
            dtype=torch.long,
            device=self.cfg.device,
        )
        with torch.no_grad():
            logits, _ = self.inference.net(tensor_obs)
            logits = logits[:, -1, :]
            masked = torch.full_like(logits, float("-inf"))
            masked[:, :5] = logits[:, :5]
            probs = F.softmax(masked, dim=-1)[:, :5]

        return {idx: probs[i] for i, idx in enumerate(neighbor_indices)}

    def get_safe_action(
        self,
        observations,
        conflict_penalty: float = 0.0,
        do_sample: bool = True,
    ) -> List[int]:
        """
        Select actions that avoid predicted conflicts with neighbors.

        Algorithm:
        1. Run one batched forward pass to get every agent's action probs.
        2. Predict each agent's most-likely next position.
        3. For each ego agent and each candidate action, check for:
           - Vertex conflict: ego and another agent occupy the same cell.
           - Edge conflict: ego and another agent swap cells.
        4. Penalize or mask conflicting actions, re-normalize, and sample.

        Args:
            observations: List of per-agent observations from the environment.
            conflict_penalty: Multiplier for conflicting action probabilities.
                0.0 = hard mask (fully remove), >0 = soft penalty.
            do_sample: If True, sample from adjusted distribution.
                If False, take argmax.

        Returns:
            List of action indices, one per agent.
        """
        num_agents = len(observations)
        all_probs = self._forward_all(observations)  # (N, 5)

        positions = [tuple(obs["global_xy"]) for obs in observations]

        # Predict each agent's most-likely next position
        predicted_actions = torch.argmax(all_probs, dim=-1).tolist()
        predicted_next = {}
        for i in range(num_agents):
            dx, dy = ACTION_TO_DELTA[predicted_actions[i]]
            predicted_next[i] = (positions[i][0] + dx, positions[i][1] + dy)

        safe_actions = []
        for ego_idx in range(num_agents):
            ego_probs = all_probs[ego_idx].clone()
            ego_pos = positions[ego_idx]

            for action_idx in range(5):
                dx, dy = ACTION_TO_DELTA[action_idx]
                ego_next = (ego_pos[0] + dx, ego_pos[1] + dy)

                for other_idx in range(num_agents):
                    if other_idx == ego_idx:
                        continue
                    # Vertex conflict
                    if ego_next == predicted_next[other_idx]:
                        if conflict_penalty == 0.0:
                            ego_probs[action_idx] = 0.0
                        else:
                            ego_probs[action_idx] *= conflict_penalty
                        break
                    # Edge conflict (swap)
                    if (
                        ego_next == positions[other_idx]
                        and predicted_next[other_idx] == ego_pos
                    ):
                        if conflict_penalty == 0.0:
                            ego_probs[action_idx] = 0.0
                        else:
                            ego_probs[action_idx] *= conflict_penalty
                        break

            # Re-normalize or fall back to wait
            total = ego_probs.sum()
            if total == 0:
                safe_actions.append(0)  # wait is safest fallback
            else:
                ego_probs = ego_probs / total
                if do_sample:
                    action = torch.multinomial(ego_probs, num_samples=1).item()
                else:
                    action = torch.argmax(ego_probs).item()
                safe_actions.append(action)

        return safe_actions

    def act(self, observations) -> List[int]:
        """
        Drop-in replacement for MAPFGPTInference.act().

        Updates internal state, then returns safe actions.
        """
        self._update_state(observations)
        return self.get_safe_action(observations)

    def act_with_info(self, observations) -> dict:
        """
        Like act(), but also returns diagnostic information.

        Returns:
            dict with keys:
                "actions": List[int] — chosen safe actions
                "probs": Tensor (N, 5) — raw policy probabilities
                "predicted_next_positions": Dict[int, tuple] — predicted
                    next position for each agent based on argmax policy
        """
        self._update_state(observations)

        all_probs = self._forward_all(observations)
        num_agents = len(observations)
        positions = [tuple(obs["global_xy"]) for obs in observations]

        predicted_actions = torch.argmax(all_probs, dim=-1).tolist()
        predicted_next = {}
        for i in range(num_agents):
            dx, dy = ACTION_TO_DELTA[predicted_actions[i]]
            predicted_next[i] = (positions[i][0] + dx, positions[i][1] + dy)

        actions = self.get_safe_action(observations)

        return {
            "actions": actions,
            "probs": all_probs,
            "predicted_next_positions": predicted_next,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_visible_neighbors(self, ego_idx: int, observations) -> List[int]:
        """Return indices of agents within agents_radius of ego."""
        ego_pos = observations[ego_idx]["global_xy"]
        r = self.cfg.agents_radius
        neighbors = []
        for i, obs in enumerate(observations):
            if i == ego_idx:
                continue
            if (
                abs(obs["global_xy"][0] - ego_pos[0]) <= r
                and abs(obs["global_xy"][1] - ego_pos[1]) <= r
            ):
                neighbors.append(i)
        return neighbors
