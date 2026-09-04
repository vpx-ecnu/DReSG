from __future__ import annotations

import dresg.utils.seed as seed_module
from dresg.utils.seed import seed_random_generators


def test_seed_random_generators_seeds_all_runtime_rngs(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        seed_module.random,
        "seed",
        lambda value: calls.append(("python", value)),
    )
    monkeypatch.setattr(
        seed_module.np.random,
        "seed",
        lambda value: calls.append(("numpy", value)),
    )
    monkeypatch.setattr(
        seed_module.torch,
        "manual_seed",
        lambda value: calls.append(("torch", value)),
    )
    monkeypatch.setattr(
        seed_module.torch.cuda,
        "manual_seed_all",
        lambda value: calls.append(("cuda", value)),
    )

    seed_random_generators(42)

    assert calls == [
        ("python", 42),
        ("numpy", 42),
        ("torch", 42),
        ("cuda", 42),
    ]
