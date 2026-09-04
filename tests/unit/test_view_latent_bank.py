from __future__ import annotations

import pytest
import torch

from dresg.models.diffusion.latents.bank import ViewLatentBank


def _bank() -> ViewLatentBank:
    return ViewLatentBank(
        {
            0: torch.zeros(1, 4, 2, 3),
            5: torch.ones(1, 4, 2, 3),
        }
    )


def test_view_latent_bank_batches_in_requested_order() -> None:
    bank = _bank()

    batch = bank.batch([5, 0])

    assert batch.shape == (2, 4, 2, 3)
    assert torch.equal(batch[0], torch.ones(4, 2, 3))
    assert torch.equal(batch[1], torch.zeros(4, 2, 3))


def test_view_latent_bank_replaces_one_controlled_batch() -> None:
    bank = _bank()
    replacement = torch.stack(
        [
            torch.full((4, 2, 3), 2.0),
            torch.full((4, 2, 3), 3.0),
        ]
    )

    bank.replace_batch([0, 5], replacement)

    assert torch.equal(bank[0], torch.full((1, 4, 2, 3), 2.0))
    assert torch.equal(bank[5], torch.full((1, 4, 2, 3), 3.0))


def test_view_latent_bank_rejects_shape_and_key_contract_violations() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        ViewLatentBank(
            {
                0: torch.zeros(1, 4, 2, 3),
                1: torch.zeros(1, 4, 3, 3),
            }
        )

    bank = _bank()
    with pytest.raises(KeyError, match="Unknown latent view"):
        bank.batch([99])
    with pytest.raises(ValueError, match="shape"):
        bank.replace_batch([0], torch.zeros(1, 4, 3, 3))



def test_view_latent_bank_rejects_autograd_state_on_construction_or_writeback() -> None:
    with pytest.raises(ValueError, match="autograd graphs"):
        ViewLatentBank({0: torch.zeros(1, 4, 2, 3, requires_grad=True)})

    bank = _bank()
    with pytest.raises(ValueError, match="autograd graphs"):
        bank.replace_batch([0], torch.zeros(1, 4, 2, 3, requires_grad=True))
