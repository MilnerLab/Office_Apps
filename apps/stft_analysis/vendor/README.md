# Vendored modules

Verbatim copies of analysis code maintained outside this repo, checked in so that
`apps/stft_analysis` runs from a clean clone with nothing but numpy/scipy/matplotlib.

**Do not hand-edit anything in this directory.** Change it upstream, then re-copy the
whole file and update the hash below. Two hand-maintained copies of the same math is a
defect in itself — an identical file fed a different constant is a different analysis,
which is exactly the bug that cost the upstream project a day (see its
`HANDOFF_pipelineB.md`, Task 11).

## fringe_core.py

Cubic-phase fringe fit: envelope fit under a pinball loss, contrast crop, Hilbert seed,
then a cubic phase refit on raw counts with covariance-propagated trust gates. Pure
numpy/scipy — no matplotlib, no file I/O, no runtime-mutated globals.

| | |
|-|-|
| upstream | `D:\Documents\University\UBC research\2026\Data\20260709\spectrometer\fringe_core.py` |
| copied | 2026-07-19 |
| sha256 | `338ca5b42764b77bb5e941e29d1ab6036d6c56fe0d489a6f9e5425ab8ab82b87` |

Verify the copy is still faithful:

```
python -c "import hashlib;print(hashlib.sha256(open('apps/stft_analysis/vendor/fringe_core.py','rb').read()).hexdigest())"
```

Consumer in this repo: `apps/stft_analysis/xcorr_fringe_fit_script.py`.

Note that the upstream file is calibrated for fringes in **wavelength** (nm). The XCORR
script does not change it — it maps its delay axis onto the pseudo-wavelength axis the
pipeline expects and maps the fitted frequency back. See that script's docstring.
