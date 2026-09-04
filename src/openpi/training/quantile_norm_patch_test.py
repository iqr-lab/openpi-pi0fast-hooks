"""Guard for a fork-local patch that is not yet in upstream openpi.

Upstream PR #971 ("Restore z-score normalization for pi0_fast_libero configs")
fixes a regression that makes the released `pi0_fast_libero` checkpoint score ~0%
on LIBERO. It is not merged upstream, so this repo carries the change locally --
see `docs/upstream_patches.md` and `patches/`.

These tests fail loudly if a future `git merge upstream/main` silently reverts the
patch. Delete this file, the patch file, and the doc entry together once upstream
merges the fix.
"""

import dataclasses
import pathlib

import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.training import config as _config


@pytest.fixture
def missing_assets_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """An assets dir with no norm stats; _load_norm_stats swallows the miss."""
    return tmp_path / "no_such_assets"


def test_override_field_exists():
    """The patch adds `use_quantile_norm_override` to DataConfigFactory."""
    assert "use_quantile_norm_override" in {f.name for f in dataclasses.fields(_config.FakeDataConfig)}


def test_override_forces_z_score(missing_assets_dir):
    """An explicit False override wins over the model-type default."""
    non_pi0 = pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180)
    assert non_pi0.model_type != _model.ModelType.PI0

    base = _config.FakeDataConfig(use_quantile_norm_override=False).create_base_config(missing_assets_dir, non_pi0)
    assert base.use_quantile_norm is False


def test_default_behaviour_unchanged(missing_assets_dir):
    """With no override, the model-type default still applies."""
    non_pi0 = pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180)
    base = _config.FakeDataConfig().create_base_config(missing_assets_dir, non_pi0)
    assert base.use_quantile_norm is True

    pi0 = pi0_config.Pi0Config()
    assert _config.FakeDataConfig().create_base_config(missing_assets_dir, pi0).use_quantile_norm is False


@pytest.mark.parametrize(
    ("config_name", "expected"),
    [
        # The two configs the patch pins. These are the regression.
        ("pi0_fast_libero", False),
        ("pi0_fast_libero_low_mem_finetune", False),
        # Untouched by the patch; guards against over-application.
        ("pi0_fast_droid", True),
        ("pi0_libero", False),
        ("pi05_libero", True),
    ],
)
def test_flag_matrix(config_name: str, expected: bool, missing_assets_dir):  # noqa: FBT001
    """Normalization flags for every LIBERO/DROID config the patch could affect."""
    train_config = _config.get_config(config_name)
    base = train_config.data.create_base_config(missing_assets_dir, train_config.model)
    assert base.use_quantile_norm is expected, (
        f"{config_name} expected use_quantile_norm={expected}. "
        "If this fails after merging upstream, the PR #971 patch was likely reverted; "
        "see docs/upstream_patches.md."
    )
