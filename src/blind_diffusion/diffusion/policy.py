from .model import UNet1D
from .diffusion import GaussianDiffusion
from .flow_matching import ConditionalOTFlowMatching


def build_action_policy(config, action_dim, condition_dim):
    dc = config["diffusion"]
    model = UNet1D(action_dim, dc["horizon"], condition_dim, base_ch=dc["base_ch"],
                   architecture=dc.get("architecture", "legacy"))
    kind = config.get("policy_type", "diffusion")
    if kind == "diffusion":
        return GaussianDiffusion(model, dc["timesteps"], dc["schedule"], dc["prediction_type"])
    if kind == "flow_matching":
        flow = config.get("flow", {})
        return ConditionalOTFlowMatching(model, flow.get("sigma_min", 1e-4), flow.get("time_scale", 1000.0))
    raise ValueError(f"Unknown policy_type: {kind}")
