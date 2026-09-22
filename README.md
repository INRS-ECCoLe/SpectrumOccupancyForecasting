# Federated Spectrum Occupancy Forecasting in an Indoor Office Environment

Measurement, forecasting and TITANIC split-learning code for an indoor
spectrum-occupancy study carried out at **INRS-EMT, Université du Québec**,
Montréal.

[![Dataset](https://img.shields.io/badge/IEEE%20DataPort-10.21227%2Ff2rv--ms26-00629B)](https://dx.doi.org/10.21227/f2rv-ms26)
[![Code](https://img.shields.io/badge/code-MIT-green)](#licence)
[![Data](https://img.shields.io/badge/data-CC%20BY%204.0-blue)](https://dx.doi.org/10.21227/f2rv-ms26)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

> **Dataset:** <https://ieee-dataport.org/documents/real-world-multi-band-rf-dataset-mass-gathering-event-spectrum-analysis-framework>
> **DOI:** <https://dx.doi.org/10.21227/f2rv-ms26>

---

## What this is

Spectrum occupancy **forecasting** — predicting what a band will look like a few
steps ahead — as opposed to **sensing**, which decides whether a band is busy
now. The corpus is one hour of ordinary working activity on an office floor,
swept with a single software-defined radio.

| | |
|---|---|
| Site | INRS-EMT office floor, Montréal, static receiver |
| Session | ~16:00–17:00 local, one hour |
| Receiver | Ettus USRP B205mini-i (AD9364), **RX2** port |
| Antenna | Dual-band 2.4/5/5.8 GHz omni, SMA male, 8 dBi (vendor) |
| Host | One commodity laptop, USB 3.0 |
| Coverage | 43 tiles, 7 bands, 25 MS/s |
| Task | 32 steps in → 8 steps out, 32 sub-channels per band |

The acquisition chain is the same one used for our outdoor mass-gathering
campaign, which is published on IEEE DataPort — see
[Citation](#citation).

---

## The headline result

**The choice of baseline decides the conclusion.** Occupancy in a
near-stationary indoor process is well predicted by a simple window mean:

| Baseline | Test MSE |
|---|---|
| Persistence (copy last observation) | 0.1914 |
| Seasonal naïve | 0.1914 |
| **Historical mean** | **0.0990** |

The window mean beats persistence by roughly a factor of two. So we report two
skill scores,

```
η_p = 1 − MSE / MSE_persistence          the conventional bar
η_b = 1 − MSE / MSE_best_baseline        the honest bar
```

and require `η_b > 0` for a model to count. **All ten forecasters clear
persistence. Only five clear the window mean.**

| Model | η_b centralized | η_b TITANIC | Δ |
|---|---|---|---|
| PatchTST | **0.0334** | 0.0223 | −0.0111 |
| Crossformer | 0.0331 | **0.0271** | −0.0061 |
| iTransformer | 0.0280 | 0.0175 | −0.0104 |
| N-HiTS | 0.0228 | 0.0097 | −0.0131 |
| DLinear | 0.0036 | 0.0144 | **+0.0109** |

Five more were evaluated identically and failed the criterion: TimesNet
(−0.083), Informer (−0.039), SCINet (−0.253), Autoformer (−0.830) and FEDformer
(−1.064). Had we reported `η_p` alone, all ten would have looked successful.

**Federation is nearly free in accuracy** — at most ~1.3 skill points, and
DLinear *improves* under TITANIC. We attribute this to TITANIC aggregating only
the input partition while downstream partitions are shared, so the averaging
step that damages FedAvg under heterogeneity touches a small fraction of the
parameters; for a model whose capacity sits in one linear map, that acts as a
regulariser.

**But not free in bandwidth.** With ρ = TITANIC payload ÷ FedAvg payload:

| Model | Params (k) | FLOPs (M) | ρ |
|---|---|---|---|
| PatchTST | 71.6 | 15.41 | 2.7 |
| Crossformer | 38.0 | 0.51 | 10.4 |
| iTransformer | 69.6 | 2.30 | 2.8 |
| N-HiTS | 293.8 | 0.58 | 0.7 |
| DLinear | 0.5 | 0.03 | 372.4 |

ρ > 1 for four of five. Activation traffic scales with batch count; weight
traffic does not. Split learning pays off when the model is large relative to
the dataset — these forecasters are tiny, so at this scale the benefit is
**memory and privacy, not bandwidth**. This matches the analysis in the TITANIC
paper itself.

---

## Repository contents

| File | Purpose |
|---|---|
| `train_spectrum_forecasting.py` | Everything: PSD → series, 10 forecasters, 3 baselines, TITANIC, per-SNR reporting, crash-safe resume |
| `make_results.py` | Regenerates the paper's tables and figures from the result CSVs |
| `collect_wifi.py` | Acquisition (shared with the sensing repo) |
| `results/` | `ALL_forecast_{summary,per_snr,complexity}.csv` |
|  `paper/` | ICASSP 4+1 LaTeX source and figures |

## Quick start

```bash
pip install numpy torch scipy h5py matplotlib

python train_spectrum_forecasting.py                    # auto-picks session
python train_spectrum_forecasting.py --quick            # smoke test
python train_spectrum_forecasting.py --models dlinear_a23,patchtst_i23
python make_results.py --csvdir results --out paper     # tables + figures
```

Rebuild the paper (needs `texlive-publishers` for `IEEEtran.cls`):

```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

Point it at your capture:

```python
ROOT        = r"...\Spectrum Collection\Indoor Office"
OUTPUT_ROOT = r"...\Spectrum Collection\Outputs"
```

---

## Method, in brief

**Series construction.** Swept dwells are stitched per band onto a uniform grid
of `C = 32` sub-channels. The sub-channel value is a *linear-domain* average of
the periodogram bins,

```
z_c(t) = 10·log10( mean_{f∈B_c} 10^(S(f,t)/10) )
```

A per-bin noise floor is the 10th percentile over the session — robust because
persistent carriers occupy the upper tail. The excess `e_c(t) = z_c(t) − N_c`
gives both the binary occupancy target and the estimated-SNR context used for
stratified reporting.

**Leakage protection.** The chronological split leaves a gap of `lookback +
horizon` between partitions, applied *within each band*. Without it the lookback
window of the first test sample overlaps the target of the last training sample.
This is silent — no error, just optimistic numbers.

**TITANIC.** The model is cut into `K` partitions on distinct clients joined by
an autograd bridge, so `f_θ = P_K ∘ … ∘ P_1` and no client holds θ in full.
Placement solves the paper's assignment programme; the constraint matrix is
totally unimodular, so the LP relaxation is integral and `scipy.linprog` gives
the exact optimum. Clients hold disjoint contiguous temporal shards.

> **What is and isn't simulated.** Partition placement, forward/backward across
> the bridge, aggregation and byte accounting are exact. There is no real
> peer-to-peer transport, so wall-clock times are compute-only and are not a
> measurement of TITANIC's real-world training speed.

---

## Metrics

MSE, MAE, RMSE (all ten papers) · sMAPE, MASE (N-HiTS / M4 protocol) · R², NMSE ·
`η_p`, `η_b` · and the occupancy view (Acc, F₁, Pd, MDR, FDR, balanced accuracy)
so a forecast can be read as an access decision.

Results are additionally stratified by the mean excess-over-noise-floor of the
forecast horizon, with a **per-bin** persistence reference so `η_p` is
meaningful inside each stratum rather than borrowed from the global average.

---

## Known limitations

State these if you build on this.

1. **Power is uncalibrated dBFS, not dBm.** Antenna gain, cable loss and RX gain
   are folded in. Relative comparisons at fixed gain only.
2. **The occupancy view is degenerate on this corpus.** The measured positive
   rate is **0.893** — the office medium is busy in the overwhelming majority of
   sub-channel slots, so an all-busy predictor scores F₁ 0.943 at balanced
   accuracy 0.5, and every model converges there. Regression metrics are
   primary; a meaningful classification study needs a stricter threshold or a
   corpus spanning quiet periods.
3. **Margins are small.** The best `η_b` is ~3%. On a near-stationary process
   there is limited predictable structure beyond the local mean, and an honest
   account should say so rather than report a large skill against a weak
   reference.
4. **Short series.** One hour at the achievable revisit period yields a few
   hundred windows per band, which limits usable model capacity.
5. **Labels are energy-detector proxies**, not verified ground truth.
6. **Single site, single receiver, one session.** A case study.

---

## Citation

If you use this code or the companion dataset, please cite:

```bibtex
@data{mahabub2026dataset,
  author    = {Mahabub, Atik and Vakili, Shervin},
  title     = {A Real-World Multi-Band {RF} Dataset from a Mass-Gathering
               Event for Spectrum Analysis Framework},
  publisher = {IEEE Dataport},
  year      = {2026},
  doi       = {10.21227/f2rv-ms26},
  url       = {https://dx.doi.org/10.21227/f2rv-ms26}
}
```

> A. Mahabub and S. Vakili, "A Real-World Multi-Band RF Dataset from a
> Mass-Gathering Event for Spectrum Analysis Framework," IEEE Dataport, 2026.
> <https://dx.doi.org/10.21227/f2rv-ms26>

*The DOI above is the outdoor mass-gathering campaign, which shares this
acquisition chain. Replace or supplement it with the indoor-office DOI once that
capture is released.*

### Key references implemented here

| Component | Reference |
|---|---|
| TITANIC | N. Su, C. Hu, B. Li, B. Li, *IEEE INFOCOM*, 2024, pp. 611–620 |
| DLinear | A. Zeng *et al.*, *AAAI*, 2023 |
| N-HiTS | C. Challu *et al.*, *AAAI*, 2023 |
| PatchTST | Y. Nie *et al.*, *ICLR*, 2023 |
| Crossformer | Y. Zhang, J. Yan, *ICLR*, 2023 |
| iTransformer | Y. Liu *et al.*, *ICLR*, 2024 |
| FedAvg | B. McMahan *et al.*, *AISTATS*, 2017 |

---

## Licence

Code: **MIT**. Data: **CC BY 4.0** via IEEE DataPort.

## Acknowledgement

Supported by NSERC grant RGPIN-2024-06358. Computational resources provided by
the Digital Research Alliance of Canada.
