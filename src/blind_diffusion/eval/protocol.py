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
