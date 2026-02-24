from pathlib import Path
from typing import Literal, Optional

import cppimport.import_hook
import torch
from huggingface_hub import hf_hub_download
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import Extra

from gpt.model import GPT, GPTConfig
from tokenizer import cost2go
from tokenizer.tokenizer import Encoder, InputParameters

# Action index -> (dr, dc) delta, matching POGEMA GridConfig.MOVES order
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


class MAPFGPTInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["MAPF-GPT"] = "MAPF-GPT"
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    path_to_weights: Optional[str] = "weights/model-6M.pt"
    device: str = "cuda"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    repo_id: str = "aandreychuk/MAPF-GPT"


def strip_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    """
    strips the given prefix from the keys in the state dictionary
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix) :]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class MAPFGPTInference:
    def __init__(self, cfg: MAPFGPTInferenceConfig, net=None):
        self.cfg: MAPFGPTInferenceConfig = cfg
        self.cost2go_data = None
        self.actions_history = None
        self.position_history = None
        self.encoder = Encoder(
            InputParameters(
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
            )
        )

        path_to_weights = Path(self.cfg.path_to_weights)
        if path_to_weights.name in ["model-2M.pt", "model-6M.pt", "model-85M.pt"]:
            hf_hub_download(
                repo_id=self.cfg.repo_id,
                filename=path_to_weights.name,
                local_dir=path_to_weights.parent,
            )
            ToolboxRegistry.info(
                f"Using weights loaded from huggingface: {path_to_weights}"
            )

        if (
            self.cfg.device in ["mps", "cuda"] and not torch.cuda.is_available()
            if self.cfg.device == "cuda"
            else not torch.backends.mps.is_available()
        ):
            ToolboxRegistry.warning(
                f"{self.cfg.device} is not available, using cpu instead!"
            )
            self.cfg.device = "cpu"

        checkpoint = torch.load(
            Path(self.cfg.path_to_weights), map_location=self.cfg.device
        )

        model_state_dict = strip_prefix_from_state_dict(checkpoint["model"])
        config_dict = checkpoint.get("model_args")
        gpt_config = GPTConfig(**config_dict)
        if net is not None:
            self.net = net
        else:
            self.net = GPT(gpt_config)
            self.net.load_state_dict(model_state_dict, strict=False)
            self.net.to(self.cfg.device)
            self.net.eval()

    def generate_input(self, observations):
        next_actions = ["" for _ in range(len(observations))]
        for agent_idx, obs in enumerate(observations):
            next_action = ""
            for m in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                new_pos = (obs["global_xy"][0] + m[0], obs["global_xy"][1] + m[1])
                if (
                    self.cost2go_data[obs["global_target_xy"]][new_pos[0]][new_pos[1]]
                    >= 0  # not wall
                    and self.cost2go_data[obs["global_target_xy"]][obs["global_xy"][0]][
                        obs["global_xy"][1]
                    ]
                    > self.cost2go_data[obs["global_target_xy"]][new_pos[0]][new_pos[1]]
                ):
                    next_action += "1"
                else:
                    next_action += "0"
            next_actions[agent_idx] = next_action

        inputs = []
        global_xy = [obs["global_xy"] for obs in observations]

        for agent_idx, obs in enumerate(observations):
            agents_info = []
            distances = []
            for j, p2 in enumerate(global_xy):
                distance = self.cost2go_data[tuple(global_xy[agent_idx])][p2[0]][p2[1]]
                if distance >= 0:
                    distances.append((j, distance))
            distances.sort(key=lambda x: (x[1], x[0]))
            sorted_agents = [agent_id for agent_id, _ in distances]
            for n in sorted_agents[: self.cfg.num_agents]:
                relative_goal = (
                    observations[n]["global_target_xy"][0] - obs["global_xy"][0],
                    observations[n]["global_target_xy"][1] - obs["global_xy"][1],
                )
                relative_xy = (
                    observations[n]["global_xy"][0] - obs["global_xy"][0],
                    observations[n]["global_xy"][1] - obs["global_xy"][1],
                )
                if (
                    -self.cfg.agents_radius <= relative_xy[0] <= self.cfg.agents_radius
                    and -self.cfg.agents_radius
                    <= relative_xy[1]
                    <= self.cfg.agents_radius
                ):
                    agents_info.append(
                        {
                            "relative_pos": relative_xy,
                            "relative_goal": relative_goal,
                            "previous_actions": self.actions_history[n],
                            "next_action": next_actions[n],
                        }
                    )
            inputs.append(
                {
                    "agents": agents_info,
                    "cost2go": cost2go.generate_cost2go_obs(
                        self.cost2go_data[obs["global_target_xy"]],
                        obs["global_xy"],
                        self.cfg.cost2go_radius,
                        self.cfg.cost2go_value_limit,
                        self.cfg.mask_cost2go,
                    ),
                }
            )

        return inputs

    def act(self, observations):
        num_agents = len(observations)
        moves = {(0, 0): "w", (-1, 0): "u", (1, 0): "d", (0, -1): "l", (0, 1): "r"}
        if self.cost2go_data is None:
            global_obs = observations[0]["global_obstacles"].copy().astype(int).tolist()
            self.cost2go_data = cost2go.precompute_cost2go(
                global_obs, self.cfg.cost2go_radius
            )

            self.actions_history = [
                ["n" for _ in range(self.cfg.num_previous_actions)]
                for _ in range(num_agents)
            ]
            self.position_history = [[obs["global_xy"]] for obs in observations]
        else:
            for i in range(num_agents):
                self.position_history[i].append(observations[i]["global_xy"])
                self.actions_history[i].append(
                    moves[
                        (
                            self.position_history[i][-1][0]
                            - self.position_history[i][-2][0],
                            self.position_history[i][-1][1]
                            - self.position_history[i][-2][1],
                        )
                    ]
                )
                self.actions_history[i] = self.actions_history[i][
                    -self.cfg.num_previous_actions :
                ]
        inputs = self.generate_input(observations)
        tensor_obs = torch.tensor(
            [self.encoder.encode(input) for input in inputs],
            dtype=torch.long,
            device=self.cfg.device,
        )

        idx_next, probs = self.net.act_with_probs(tensor_obs, do_sample=False)
        actions = torch.squeeze(idx_next).tolist()

        if not isinstance(actions, list):
            actions = [actions]
        # probs shape: (num_agents, 5) — stored for subclasses that need per-agent distributions
        self._last_probs = probs  # (num_agents, 5), on self.cfg.device
        return actions

    def reset_states(self):
        self.cost2go_data = None
        self.actions_history = None
        self.position_history = None


class MAPFGPTWithPIBTConfig(MAPFGPTInferenceConfig):
    name: Literal["MAPF-GPT-PIBT"] = "MAPF-GPT-PIBT"


class MAPFGPTWithPIBT(MAPFGPTInference):
    """
    MAPF-GPT + PIBT post-processing for collision-free actions.

    The GPT model proposes a preferred action per agent.  PIBT then resolves
    conflicts: agents are processed in descending distance-to-goal order
    (furthest = highest priority), and each agent tries its GPT-suggested
    action first before falling back to cost-to-go ordering.
    """

    def __init__(self, cfg: MAPFGPTWithPIBTConfig, net=None):
        super().__init__(cfg, net)

    def _dist(self, target_xy, pos):
        return self.cost2go_data[target_xy][pos[0]][pos[1]]

    def _pibt(self, agent_idx, observations, gpt_probs, priority_order, occupied_by, reserved, actions):
        """
        Assign a collision-free action to agent_idx.

        Candidates are drawn from the GPT probability distribution: actions
        whose target cell is already reserved or is a wall are masked to zero,
        then we pick in descending probability order.  Priority inheritance
        recursively asks lower-priority occupants to yield before masking.
        """
        pos = tuple(observations[agent_idx]["global_xy"])
        target = tuple(observations[agent_idx]["global_target_xy"])

        # Work with a mutable copy of this agent's probability vector (cpu, numpy-free).
        probs = gpt_probs[agent_idx].clone()  # shape (5,), on device

        # Zero out wall / OOB actions upfront.
        for a in range(len(ACTIONS)):
            dr, dc = ACTIONS[a]
            npos = (pos[0] + dr, pos[1] + dc)
            if a != 0 and self._dist(target, npos) < 0:
                probs[a] = 0.0

        while probs.sum() > 0:
            action_idx = int(probs.argmax())
            dr, dc = ACTIONS[action_idx]
            npos = (pos[0] + dr, pos[1] + dc)

            if npos in reserved:
                # Cell claimed by a higher-priority agent — mask and try next.
                probs[action_idx] = 0.0
                continue

            occupant = occupied_by.get(npos)
            if occupant is not None and occupant != agent_idx:
                if priority_order[occupant] < priority_order[agent_idx]:
                    if actions[occupant] is not None:
                        # Occupant already decided — check if it vacated npos.
                        odr, odc = ACTIONS[actions[occupant]]
                        opos = tuple(observations[occupant]["global_xy"])
                        if (opos[0] + odr, opos[1] + odc) != npos:
                            # Occupant moved away — claim npos.
                            actions[agent_idx] = action_idx
                            reserved.add(npos)
                            return True
                        # Occupant stays — mask this action and try next.
                        probs[action_idx] = 0.0
                        continue
                    # Priority inheritance: ask occupant to yield.
                    reserved.add(npos)
                    success = self._pibt(occupant, observations, gpt_probs, priority_order, occupied_by, reserved, actions)
                    if success:
                        actions[agent_idx] = action_idx
                        reserved.add(npos)
                        return True
                    # Occupant couldn't move — mask and try next.
                    reserved.discard(npos)
                    probs[action_idx] = 0.0
                    continue
                else:
                    # Higher-priority occupant — mask and try next.
                    probs[action_idx] = 0.0
                    continue

            # Cell is free — claim it.
            actions[agent_idx] = action_idx
            reserved.add(npos)
            return True

        # All actions exhausted — stay in place.
        actions[agent_idx] = 0
        reserved.add(pos)
        return False

    def act(self, observations):
        # 1. Run GPT to get preferred actions and per-agent probability distributions.
        gpt_actions = super().act(observations)
        gpt_probs = self._last_probs  # (num_agents, 5), on self.cfg.device

        num_agents = len(observations)

        # 2. Assign priorities: furthest from goal = highest priority.
        distances = []
        for obs in observations:
            pos = tuple(obs["global_xy"])
            target = tuple(obs["global_target_xy"])
            d = self._dist(target, pos)
            distances.append(d if d >= 0 else 0)

        priority_order = {i: (distances[i], -i) for i in range(num_agents)}
        sorted_agents = sorted(range(num_agents), key=lambda i: priority_order[i], reverse=True)

        occupied_by = {tuple(obs["global_xy"]): i for i, obs in enumerate(observations)}

        actions = [None] * num_agents
        reserved = set()

        for agent_idx in sorted_agents:
            if actions[agent_idx] is not None:
                continue
            self._pibt(agent_idx, observations, gpt_probs, priority_order, occupied_by, reserved, actions)

        final_actions = [a if a is not None else 0 for a in actions]

        # Log any agent whose final action differs from GPT's suggestion.
        for i, (gpt_a, final_a) in enumerate(zip(gpt_actions, final_actions)):
            if gpt_a != final_a:
                ToolboxRegistry.info(
                    f"PIBT override: agent {i} at {observations[i]['global_xy']} "
                    f"redirected from action {gpt_a} to {final_a}"
                )

        return final_actions


class PIBTInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["PIBT"] = "PIBT"


class PIBTInference:
    """
    Basic PIBT (Priority Inheritance with Backtracking) implementation.

    Priority is assigned by distance-to-goal from the precomputed cost2go map:
    agents farther from their goal receive higher priority.

    Each timestep:
      1. Compute cost-to-go distance for every agent.
      2. Sort agents by descending distance (furthest = highest priority).
      3. Process agents in priority order via _pibt(), which recursively
         handles priority inheritance when a desired cell is occupied by a
         lower-priority agent.
    """

    def __init__(self, cfg: PIBTInferenceConfig):
        self.cfg = cfg
        self.cost2go_data = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dist(self, target_xy, pos):
        """Cost-to-go distance from pos to target_xy; negative means wall/OOB."""
        return self.cost2go_data[target_xy][pos[0]][pos[1]]

    def _pibt(self, agent_idx, observations, priority_order, occupied_by, reserved, actions):
        """
        Assign an action to agent_idx using PIBT with priority inheritance.

        Parameters
        ----------
        agent_idx    : index of the agent being processed
        observations : list of per-agent observation dicts
        priority_order : dict mapping agent_idx -> priority value (higher = more important)
        occupied_by  : dict mapping grid cell (r,c) -> agent index currently there
        reserved     : set of (r,c) cells already claimed for the next timestep
        actions      : output list (length == num_agents), filled in place

        Returns True if the agent found a non-colliding action.
        """
        pos = tuple(observations[agent_idx]["global_xy"])
        target = tuple(observations[agent_idx]["global_target_xy"])

        # Sort candidate actions by cost-to-go of the resulting cell
        # (ascending = closer to goal = preferred), walls last, wait last of all.
        def move_cost(action_idx):
            dr, dc = ACTIONS[action_idx]
            npos = (pos[0] + dr, pos[1] + dc)
            d = self._dist(target, npos)
            if d < 0:
                return float("inf")  # wall / OOB
            return d

        candidate_actions = sorted(range(len(ACTIONS)), key=move_cost)

        for action_idx in candidate_actions:
            dr, dc = ACTIONS[action_idx]
            npos = (pos[0] + dr, pos[1] + dc)

            # Skip walls / OOB for non-wait actions.
            if self._dist(target, npos) < 0 and action_idx != 0:
                continue

            # Cell already claimed by a strictly higher-priority agent — skip.
            if npos in reserved:
                continue

            # Check if a lower-priority agent currently occupies npos.
            occupant = occupied_by.get(npos)
            if occupant is not None and occupant != agent_idx:
                if priority_order[occupant] < priority_order[agent_idx]:
                    # If the occupant already has an assigned action this step,
                    # check whether it has vacated npos.
                    if actions[occupant] is not None:
                        # Occupant was already processed; it stays at npos if
                        # its action is wait (action 0 means npos == its pos).
                        # If it moved elsewhere, npos is free.
                        odr, odc = ACTIONS[actions[occupant]]
                        opos = tuple(observations[occupant]["global_xy"])
                        if (opos[0] + odr, opos[1] + odc) != npos:
                            # Occupant moved away — npos is free.
                            actions[agent_idx] = action_idx
                            reserved.add(npos)
                            return True
                        # Occupant stays at npos — try next candidate.
                        continue
                    # Priority inheritance: ask the occupant to yield.
                    reserved.add(npos)  # tentatively block npos
                    success = self._pibt(occupant, observations, priority_order, occupied_by, reserved, actions)
                    if success:
                        # Occupant moved away — claim npos.
                        actions[agent_idx] = action_idx
                        reserved.add(npos)
                        return True
                    # Occupant could not move — release and try next candidate.
                    reserved.discard(npos)
                    continue
                else:
                    # Higher-priority occupant — cannot displace it.
                    continue

            # Cell is free; claim it.
            actions[agent_idx] = action_idx
            reserved.add(npos)
            return True

        # No valid move found — stay in place.
        actions[agent_idx] = 0
        reserved.add(pos)
        return False

    # ------------------------------------------------------------------
    # Public interface (matches MAPFGPTInference)
    # ------------------------------------------------------------------

    def act(self, observations):
        num_agents = len(observations)

        if self.cost2go_data is None:
            global_obs = observations[0]["global_obstacles"].copy().astype(int).tolist()
            self.cost2go_data = cost2go.precompute_cost2go(global_obs, 5)

        # Distance-to-goal for each agent (treat unreachable as 0).
        distances = []
        for obs in observations:
            pos = tuple(obs["global_xy"])
            target = tuple(obs["global_target_xy"])
            d = self._dist(target, pos)
            distances.append(d if d >= 0 else 0)

        # Priority value per agent: (distance, -agent_idx) so that ties are
        # broken deterministically in favour of lower-indexed agents.
        priority_order = {i: (distances[i], -i) for i in range(num_agents)}

        # Sort agents by descending priority (furthest from goal goes first).
        sorted_agents = sorted(range(num_agents), key=lambda i: priority_order[i], reverse=True)

        # Map each occupied cell to the agent index currently there.
        occupied_by = {tuple(obs["global_xy"]): i for i, obs in enumerate(observations)}

        actions = [None] * num_agents  # None = not yet assigned
        reserved = set()

        for agent_idx in sorted_agents:
            if actions[agent_idx] is not None:
                # Already assigned via priority inheritance from a higher-priority agent.
                continue
            self._pibt(agent_idx, observations, priority_order, occupied_by, reserved, actions)

        # Replace any remaining None with wait (shouldn't happen, but be safe).
        return [a if a is not None else 0 for a in actions]

    def reset_states(self):
        self.cost2go_data = None
