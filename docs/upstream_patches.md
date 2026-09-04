# Fork-local patches

Changes this fork carries that are **not** in `Physical-Intelligence/openpi`, and that
must survive future `git merge upstream/main`.

Each entry lists what it does, why, how to verify it is still applied, and the condition
under which it should be deleted.

## After every upstream merge

```bash
git log --grep='PATCH(upstream-pending)' --oneline    # list the patch commits
uv run pytest src/openpi/training/quantile_norm_patch_test.py
```

A merge can revert one of these silently -- git resolves in upstream's favour without a
conflict when only one side "changed" a line. The guard tests exist to make that loud.
If a guard fails, re-apply the corresponding patch file:

```bash
git apply patches/0001-restore-z-score-norm-pi0-fast-libero.patch
```

---

## 0001 -- Restore z-score normalization for `pi0_fast_libero` configs

- **Upstream PR:** https://github.com/Physical-Intelligence/openpi/pull/971 (open, not merged)
- **Upstream issue:** https://github.com/Physical-Intelligence/openpi/issues/849
- **Patch file:** `patches/0001-restore-z-score-norm-pi0-fast-libero.patch`
- **Guard test:** `src/openpi/training/quantile_norm_patch_test.py`
- **Touches:** `src/openpi/training/config.py` (+13/-1)

### Why

Evaluating the released `pi0_fast_libero` checkpoint on upstream `main` gives ~0% success
on LIBERO -- the arm spins in place.

At commit `e4580662`, `LeRobotLiberoDataConfig` built its `DataConfig` without setting
`use_quantile_norm`, so the pi0-FAST LIBERO configs trained and served with **z-score**
normalization. Later upstream commits made `DataConfigFactory.create_base_config`
force-set `use_quantile_norm = model_type != PI0`, which silently flipped
`pi0_fast_libero` and `pi0_fast_libero_low_mem_finetune` to quantile normalization --
incompatible with the released checkpoint.

`e4580662` is this fork's merge base with upstream, so the regression arrived with the
merge in `8fb052d`. That is why this patch became necessary at that point and not before.

### What it changes

- `DataConfigFactory` gains `use_quantile_norm_override: bool | None = None`. The default
  `None` preserves the model-type default for every existing config.
- `pi0_fast_libero` and `pi0_fast_libero_low_mem_finetune` pin the override to `False`.

Resulting flag matrix (asserted by the guard test):

| config | `use_quantile_norm` |
| --- | --- |
| `pi0_fast_libero` | False |
| `pi0_fast_libero_low_mem_finetune` | False |
| `pi0_fast_droid` | True |
| `pi0_libero` | False |
| `pi05_libero` | True |

### Caveat

Anyone who finetuned pi0-FAST on LIBERO against upstream `main` trained under *quantile*
normalization. For those checkpoints this override makes the choice explicit and
per-config rather than a silent default flip -- but it does mean such a checkpoint needs
`use_quantile_norm_override=True` (or its own config) to serve correctly.

### Remove when

Upstream merges PR #971. The next `git merge upstream/main` will most likely conflict on
these lines -- two versions of the same change, which is the loud outcome we want. Resolve
by taking upstream's version, then delete: this section, `patches/0001-*.patch`, and
`src/openpi/training/quantile_norm_patch_test.py`.
