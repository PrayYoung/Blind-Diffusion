import torch


def previous_actions(actions: torch.Tensor) -> torch.Tensor:
    """Align action[t - 1] with observation[t], using zero before obs[0]."""
    return torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)


def compute_pre_action_beliefs(wm, obs: torch.Tensor, actions: torch.Tensor,
                              deterministic: bool = False, blackout=None) -> torch.Tensor:
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
    if blackout is None:
        blackout = torch.zeros(obs.shape[:2], dtype=torch.bool, device=obs.device)
    if blackout.shape != obs.shape[:2] or blackout[:, 0].any():
        raise ValueError("Blackout must have shape [B,T] and keep the first observation visible")
    beliefs = []
    for timestep in range(sequence_length):
        posterior = wm.rssm.observe_step(obs_embed[:, timestep], action_inputs[:, timestep], state,
                                       deterministic=deterministic)
        if blackout[:, timestep].any():
            prior = wm.rssm.imagine_step(action_inputs[:, timestep], state, deterministic=deterministic)
            state = {k: torch.where(blackout[:, timestep, None], prior[k], posterior[k]) for k in ("h", "z")}
        else:
            state = posterior
        beliefs.append(torch.cat([state["h"], state["z"]], dim=-1))
    return torch.stack(beliefs, dim=1)


def sample_chunk_starts(valid, is_start, horizon, burn_in):
    """Require real first-action labels and warm non-initial windows."""
    length = valid.shape[1]
    if horizon > length:
        raise ValueError("Action horizon exceeds sequence length")
    positions = torch.arange(length - horizon + 1, device=valid.device)
    allowed = valid[:, :len(positions)] & (is_start[:, None] | (positions >= burn_in))
    if not allowed.any(dim=1).all():
        raise ValueError("No valid action chunk after burn-in")
    return torch.multinomial(allowed.float(), 1).squeeze(1)


def sample_blackout(batch_size, length, device, probability=0.5, warmup=5, max_duration=20):
    mask = torch.zeros(batch_size, length, device=device, dtype=torch.bool)
    if length <= warmup:
        return mask
    for b in range(batch_size):
        if torch.rand((), device=device) < probability:
            start = int(torch.randint(max(1, warmup), length, (), device=device))
            duration = int(torch.randint(1, max_duration + 1, (), device=device))
            mask[b, start:min(length, start + duration)] = True
    return mask


def sample_conditioning_queries(valid, is_start, horizon, burn_in, probability=0.5,
                                warmup=5, max_duration=20):
    """Select chunks first, then ensure augmented queries are actually blind.

    Feasible blind depths are sampled uniformly. Sequences with no post-warmup
    real label stay clean; the initial observation is always visible.
    """
    if warmup < 1 or max_duration < 1 or not 0 <= probability <= 1:
        raise ValueError("Invalid query blackout configuration")
    starts = sample_chunk_starts(valid, is_start, horizon, burn_in)
    mask = torch.zeros_like(valid, dtype=torch.bool)
    for b in range(len(starts)):
        if torch.rand((), device=valid.device) >= probability:
            continue
        positions = torch.arange(valid.shape[1] - horizon + 1, device=valid.device)
        lower = max(warmup, 0 if is_start[b] else burn_in)
        eligible = positions[(positions >= lower) & valid[b, :len(positions)]]
        if not len(eligible):
            continue
        query = eligible[torch.randint(len(eligible), (), device=valid.device)]
        starts[b] = query
        depth = int(torch.randint(1, min(max_duration, int(query) - warmup + 1) + 1, (), device=valid.device))
        mask[b, int(query) - depth + 1:int(query) + 1] = True
    return starts, mask
