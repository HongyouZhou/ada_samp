def build_timestep_sampler(cfg):
    typ = cfg.get("type", "uniform")
    T = cfg.get("T", 1000)
    if typ == "uniform":
        from .timestep_sampler import UniformTimestepSampler
        return UniformTimestepSampler(T)
    elif typ == "linear":
        from .timestep_sampler import LinearTimestepSampler
        return LinearTimestepSampler(T)
    elif typ == "importance":
        from .timestep_sampler import ImportanceTimestepSampler
        return ImportanceTimestepSampler(T)
    else:
        raise ValueError(f"Unknown timestep sampler: {typ}")


def build_noise_sampler(cfg):
    typ = cfg.get("type", "gaussian")
    if typ == "gaussian":
        from .noise_sampler import GaussianNoiseSampler
        return GaussianNoiseSampler()
    elif typ == "uniform":
        from .noise_sampler import UniformNoiseSampler
        return UniformNoiseSampler()
    elif typ == "rademacher":
        from .noise_sampler import RademacherNoiseSampler
        return RademacherNoiseSampler()
    else:
        raise ValueError(f"Unknown noise sampler: {typ}")