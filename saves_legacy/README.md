# `saves_legacy` — superseded runs, kept for the record

**Do not compare anything in here with anything in `saves/`, or with the other generation in
here.** Each generation was produced by a materially different pipeline. They are archived
because they are the evidence behind `FINDINGS.md`, not because they are still usable numbers.

They also had to leave `saves/`: `train_saver._check_done` treats *any* legacy-layout CSV as
"this cell is done", and `legacy_dir()` keys on (dataset, network, loss, datasize, batch_size)
with **no seed, lr or epoch count**. While these directories sat in `saves/`, every re-run —
including every new seed and every new learning rate — was silently skipped. Verified: all 8
sampled cells returned `previously_done=True` before the move and `False` after.

## `gen1_original/` — 90 runs

The original committed grid. CIFAR-10 (30) + ImageNet64 val on two backbones (60).

Reconstructed here from two places: the `LPIPS`/`MSSIM` cells had been archived when those
losses were fixed, and the other four losses were still live in `saves/`. Together they are the
complete original grid again. `ORIGINAL_README.md` is the note written at that first archiving.

Produced with, all of which are now known-wrong or since changed:

- **`MSSIM` was a near-no-op** — `pytorch_msssim.MS_SSIM(win_size=1)`, ~300× too weak a
  gradient (`FINDINGS.md` B1).
- **`LPIPS` got the wrong input range** — `normalize=False` fed `[0,1]` data (B2).
- **Unseeded** — network init and shuffle order free-running, so cells differ by init as well as
  by the variable under test (B5).
- **Old probe** — no feature standardisation, two-way split, no `Linear` probe.
- Old directory layout `{LOSS}-{datasize}-BS{bs}`, no `config.json`, no checkpoints.

The CIFAR half sat under `saves/CIFAR/` while the dataset key was `CIFAR_10`, so skip-if-exists
never matched it (`TODO.md` T1.0).

## `gen2_lossfix_oldprobe/` — 35 runs

CIFAR-10 `conv_big_z`, re-run after B1/B2 were fixed. **Seeded** (seed 42 throughout), so cells
differing only in loss share an identical initialisation — this is the grid behind the
`LPIPS` vs `LPIPS1` paired comparison in `FINDINGS.md` §3, and that comparison is internally
valid because both arms are from this generation.

Superseded because it still used the **old probe**: no `StandardScaler`, two-way split, no
`Linear` classifier. That matters more than it sounds. Re-probing an untrained `conv_big_z`
with the current scaled probe gives **MLP ≈ 0.399**, against the **0.167** recorded at epoch 0
here. The unscaled probe was understating the random-feature floor by ~0.23, which means every
"training improved the representation" claim measured against it is inflated.

## What is safe to still cite

- **Within-generation, same-probe comparisons.** `LPIPS` vs `LPIPS1` in gen2. The relative shape
  of the loss × datasize grid within gen1.
- **Collapse behaviour**, which is measured from the network's own output variance and so does
  not depend on the probe at all: three `DISTS` cells collapsed in gen1, and the pattern
  rearranged completely in gen2 under nothing but a seed change (`FINDINGS.md` B3).

## What is not safe

- Any absolute accuracy, against anything in `saves/`.
- Any baseline claim resting on the epoch-0 number — it is the unscaled-probe floor.
- gen1 vs gen2 on `MSSIM`/`LPIPS` as an architecture or data effect: the loss itself changed.
