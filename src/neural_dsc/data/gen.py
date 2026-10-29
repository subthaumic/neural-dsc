import torch


def sample_quadrant_data(n_total=1000, low=-3, high=3, generator=None):
    """Draw labelled quadrant points from a uniform box.

    Passing a local generator supports a reproducible online training stream
    without changing PyTorch's process-global random state.
    """
    X = (high - low) * torch.rand(n_total, 2, generator=generator) + low
    x_nonnegative = X[:, 0] >= 0
    y_nonnegative = X[:, 1] >= 0
    y = torch.empty(n_total, dtype=torch.long)
    y[x_nonnegative & y_nonnegative] = 0
    y[~x_nonnegative & y_nonnegative] = 1
    y[~x_nonnegative & ~y_nonnegative] = 2
    y[x_nonnegative & ~y_nonnegative] = 3
    return X, y


# Data generation submodule for quadrants data
def generate_quadrant_data(n_total=1000, seed=42, low=-3, high=3):
    generator = torch.Generator().manual_seed(seed)
    return sample_quadrant_data(
        n_total=n_total,
        low=low,
        high=high,
        generator=generator,
    )
