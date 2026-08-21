import torch


def previous_actions(actions: torch.Tensor) -> torch.Tensor:
    """Align action[t - 1] with observation[t], using zero before obs[0]."""
    return torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)


def compute_pre_action_beliefs(wm, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return the belief after obs[t] but before actions[t] is applied.

    Evaluation observes the current observation using the preceding executed
    action. Reproduce that causal convention in offline diffusion training so
    a target action chunk cannot leak its first action into its conditioning.
    """
    obs_embed = wm.encoder(obs)
    batch_size, sequence_length, _ = obs.shape
    state = {
        "h": torch.zeros(batch_size, wm.rssm.deter_dim, device=obs.device, dtype=obs.dtype),
        "z": torch.zeros(batch_size, wm.rssm.stoch_dim, device=obs.device, dtype=obs.dtype),
    }
    action_inputs = previous_actions(actions)
    beliefs = []
    for timestep in range(sequence_length):
        state = wm.rssm.observe_step(obs_embed[:, timestep], action_inputs[:, timestep], state)
        beliefs.append(torch.cat([state["h"], state["z"]], dim=-1))
    return torch.stack(beliefs, dim=1)
