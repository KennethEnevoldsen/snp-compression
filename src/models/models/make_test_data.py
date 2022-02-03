import torch


def make_test_data(
    hidden_dim=128, types=10, output_dim=10_000, noise=0.1, NA: Union[int, None] = 0
):
    import random

    def add_noise(x):
        for i, x in enumerate(x):
            if random.uniform(0, 1) < noise:
                x[i] = NA

    def make_random_snp(length=output_dim // 128):
        start = random.randint(0, 10)
        step = random.randint(1, 10)
        stop = length * step + start
        return [t % 3 for t in range(start, stop, step)]

    hidden_vars = random.choices(range(types), k=hidden_dim)
    patterns = {t: make_random_snp() for t in range(types)}
    input_ = [v for hv in hidden_vars for v in patterns[hv]]
    input_ += (output_dim - len(input_)) * [0]
    return torch.Tensor(input_)
