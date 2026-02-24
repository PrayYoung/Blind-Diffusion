from typing import List, Dict
import numpy as np


def summarize_episode_metrics(episodes: List[Dict]) -> Dict[str, float]:
    success = [e.get("success", 0.0) for e in episodes]
    collision = [e.get("collision", 0.0) for e in episodes]
    return {
        "success_rate": float(np.mean(success)) if success else 0.0,
        "collision_rate": float(np.mean(collision)) if collision else 0.0,
    }
