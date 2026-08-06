# Pre-loss-fix runs (archived, do not compare against `saves/`)

The 30 `LPIPS-*` and `MSSIM-*` runs that were in `saves/` before the loss bugs B1 and B2
(see `../FINDINGS.md`) were fixed. Moved here rather than deleted so the old numbers survive.

They were produced with:

- **`MSSIM`** = `pytorch_msssim.MS_SSIM(win_size=1)` — a near-no-op. The single-pixel Gaussian
  window destroys the structural term; measured ~300x weaker gradient than real SSIM for the
  same distortion. These runs do not measure multi-scale SSIM.
- **`LPIPS`** = `LearnedPerceptualImagePatchSimilarity(normalize=False)` fed `[0, 1]` data,
  i.e. LPIPS assuming its input is already `[-1, 1]`. Half the intended dynamic range.

The LPIPS behaviour has been kept as a first-class loss named **`LPIPS1`**, so the effect of
the fix is measurable directly in `saves/` — `LPIPS` vs `LPIPS1` under an identical seed.
Prefer that comparison to these archived runs, which are additionally unseeded (n=1, free
-running init, ±0.032 noise floor on CIFAR MLP).

Kept out of `saves/` on purpose: `plots/plot_training_runs.py` globs `saves/*/*/*/` and would
otherwise pick these up and plot them as if they were current.
