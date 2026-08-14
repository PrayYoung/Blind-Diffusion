from typing import List, Dict
import numpy as np


def summarize_episode_metrics(episodes: List[Dict]) -> Dict[str, float]:
    success = [e.get("success", 0.0) for e in episodes]
    collision = [e.get("collision", 0.0) for e in episodes]
    returns = [e["return"] for e in episodes if "return" in e]
    lengths = [e["length"] for e in episodes if "length" in e]
    metrics = {
        "num_episodes": int(len(episodes)),
        "success_count": int(sum(success)),
        "collision_count": int(sum(collision)),
        "success_rate": float(np.mean(success)) if success else 0.0,
        "collision_rate": float(np.mean(collision)) if collision else 0.0,
    }
    if returns:
        metrics["avg_return"] = float(np.mean(returns))
    if lengths:
        metrics["avg_length"] = float(np.mean(lengths))
    return metrics
