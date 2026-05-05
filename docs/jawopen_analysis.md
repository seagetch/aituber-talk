# SadTalker jawOpen Source Analysis

Date: 2026-05-05

SadTalker `coeff_3dmm[:, :64]` values are 3DMM expression coefficients, not ARKit blendshape coefficients. Therefore there is no authoritative `jawOpen` column in the SadTalker output. The VMC `jawOpen` value must be derived from one of the expression coefficients.

Local analysis generated SadTalker coefficients for three bundled speech samples:

- `SadTalker/examples/driven_audio/japanese.wav`
- `SadTalker/examples/driven_audio/chinese_news.wav`
- `SadTalker/examples/driven_audio/RD_Radio31_000.wav`

For each sample, audio RMS was resampled to the coefficient frame count. Each expression index was scored by how strongly it separated high-RMS speech frames from low-RMS frames, with lag search from -8 to +8 frames.

Top stable positive speech-correlated candidates:

| index | sign consistency | mean abs effect | effects by sample |
|---:|---:|---:|---|
| 35 | 1.00 | 0.840 | +1.300, +0.607, +0.615 |
| 17 | 1.00 | 0.767 | +0.804, +0.704, +0.793 |
| 39 | 1.00 | 0.741 | +0.870, +0.760, +0.594 |
| 19 | 1.00 | 0.654 | +0.631, +0.763, +0.567 |

Previously assumed candidates were not stable:

| index | sign consistency | mean abs effect | effects by sample |
|---:|---:|---:|---|
| 24 | 0.33 | 1.096 | +1.317, +1.098, -0.872 |
| 29 | 0.33 | 1.085 | +1.035, +1.209, -1.010 |
| 37 | 0.33 | 1.085 | -1.484, -1.066, +0.706 |
| 50 | 0.33 | 1.088 | +1.495, +1.029, -0.740 |

Conclusion: use expression index `35` as the default source for VMC `jawOpen`. Keep the source index configurable because the SadTalker expression basis is not semantically labeled and the final target model may prefer a different derived signal.

Generated analysis artifacts live under `tmp_jawopen_analysis/` and include `rms_coeff_correlation.csv` plus generated coefficient `.npy` files.
