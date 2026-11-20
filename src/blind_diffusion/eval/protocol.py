def is_blackout(step, warmup=5, period=20, duration=0):
    if period <= 0 or not 0 <= duration <= period or warmup < 1:
        raise ValueError("Blackout requires warmup>=1, period>0 and 0<=duration<=period")
    return step >= warmup and (step - warmup) % period < duration


def task_success(env, info=None):
    """Use the environment's task predicate; reward may be shaped."""
    if hasattr(env, "is_success"):
        result = env.is_success()
        return bool(result.get("task", False)) if isinstance(result, dict) else bool(result)
    return bool((info or {}).get("success", False))


def object_observation_sham(obs, offset):
    """Shift observed cube position and its relative feature, not physics."""
    import numpy as np
    delta = np.asarray(offset, dtype=np.float32)
    if delta.shape != (3,):
        raise ValueError("Object sham offset must have three coordinates")
    if not np.any(delta):
        return obs
    object_state = np.asarray(obs["object"]).copy()
    if object_state.shape != (10,):
        raise ValueError("Object sham expects Lift position/quaternion/eef-relative observations")
    object_state[:3] += delta
    object_state[7:10] -= delta
    return {**obs, "object": object_state}
