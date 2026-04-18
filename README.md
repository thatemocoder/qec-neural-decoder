# Neural Network Decoders for Quantum Error Correction

**Comparing MLP, LSTM, Transformer, CNN, and GNN decoders against classical MWPM for surface code QEC — from a textbook depolarising noise model through to a real IBM quantum hardware circuit with ACES calibration noise.**

Built during a research internship at the University of Sydney with Robin Harper (Quantum Theory Group).

---

## What this project is

Quantum computers make errors. Before any useful algorithm can run, those errors must be corrected in real time — a classical post-processing step called **decoding**. The standard decoder is **MWPM (Minimum Weight Perfect Matching)**, a graph algorithm that is near-optimal but computationally expensive. This project asks: can neural networks learn to decode a surface code, and how close can they get?

The work runs in two phases.

**Phase 1** trains five NN architectures (MLP, LSTM, Transformer, CNN, GNN) on a simulated distance-3 rotated surface code under depolarising noise, using [Stim](https://github.com/quantumlib/Stim) for circuit simulation and [PyMatching](https://github.com/oscarhiggott/PyMatching) as the MWPM reference. Decoders are compared across noise levels, code distances (d=3/5/7), and training regimes including multi-noise training.

**Phase 2** moves to a real hardware circuit — a d=5 HeavyHex surface code on 65 physical qubits using IBM Heron r2/r3 connectivity, with a full ACES noise model calibrated from actual ibm_boston device data. Circuits were provided by Robin Harper.

---

## Results

### Phase 1 — d=3 rotated surface code, p=0.001

| Decoder | LER | vs Trivial | Params |
|---|---|---|---|
| Trivial (always predict 0) | 5.45% | — | — |
| MLP — single noise | 1.45% | 3.8× | 60,801 |
| LSTM — single noise | 1.09% | 5.0× | 20,545 |
| **CNN spatial** | **0.87%** | **6.3×** | ~4k |
| Transformer | 2.30% | 2.4× | ~8k |
| GNN | 19.1% | 0.3× | ~3k |
| MLP — multi noise | 15.1% | 0.4× | 60,801 |
| LSTM — multi noise | 14.6% | 0.4× | 20,545 |
| **MWPM (classical ref.)** | **0.23%** | **23.9×** | — |

The CNN spatial decoder is the best neural result at 6.3×. The LSTM achieves 5× with 3× fewer parameters than the MLP. MWPM remains ~4× better than the best NN at this scale. Multi-noise training failed — insufficient training data per noise level given the 100k total budget.

### Phase 2 — d=5 HeavyHex, real ACES noise, 9 syndrome rounds

| Decoder | LER | vs Trivial |
|---|---|---|
| Trivial | 50.1% | — |
| MLP / LSTM / Transformer | ~49–50% | ~1.0× |
| **MWPM (classical ref.)** | **22.5%** | **2.2×** |

All NN decoders failed to learn on the real hardware circuit. Only MWPM succeeded. The most likely cause is insufficient training data: 70k shots is far too few for a model to learn the highly correlated, non-uniform ACES error distribution. This result directly motivates the sample efficiency analysis in notebook 18 and sets the research agenda for further work.

---

## Notebook guide

### Phase 1 — Core pipeline (run in order, ~15 min)

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_generation` | Stim d=3 circuit, 100k shots, detection events + raw measurements, 6-level sweep data |
| 02 | `02_decoder_training` | MLP on detection events (P1) and raw measurements (P2) — shared labels, fair comparison |
| 03 | `03_noise_sweep` | Both MLPs evaluated across p=0.0005 to p=0.02 |
| 04 | `04_pymatching_benchmark` | MWPM reference with Wilson CI error bars |
| 05 | `05_lstm_decoder` | LSTM processing syndromes round-by-round (9 rounds × 8 stabilisers) |
| 06 | `06_comparison_figures` | All 6 presentation figures |

### Phase 1 — Extensions

| # | Notebook | What it does |
|---|---|---|
| 01B | `01B_multi_noise_data` | Equal samples from all 6 noise levels |
| 02B | `02B_multi_noise_training` | Retrains MLP + LSTM on the mixed dataset |
| 03B | `03B_multi_noise_sweep` | Single-noise vs multi-noise sweep comparison |
| 07 | `07_distance_scaling_data` | Circuits and sweep data for d=3, 5, 7 |
| 08 | `08_distance_scaling_train` | LSTM + Transformer at each distance; scaling plot |
| 10 | `10_transformer_decoder` | Self-attention over syndrome rounds; attention map visualisation |
| 11 | `11_gnn_decoder` | Message-passing GNN on stabiliser Tanner graph |
| 12 | `12_cnn_decoder` | 2D conv on stabiliser grid; learned filter visualisation |
| 13 | `13_grand_comparison` | All decoders in one final figure; params vs LER scatter |

### Phase 2 — Robin's HeavyHex circuits (run in order, ~60 min)

| # | Notebook | What it does |
|---|---|---|
| 14 | `14_heavyhex_data_mwpm` | Loads circuit via version-safe extraction. 100k shots. MWPM baseline. |
| 15 | `15_heavyhex_decoders` | MLP (flat 240), LSTM-det (11×24), LSTM-raw (11×40), Transformer |
| 16 | `16_decay_curve` | LER vs syndrome rounds; exponential fit for ε_L per round |
| 17 | `17_speed_benchmark` | Throughput (shots/sec) and latency across batch sizes |
| 18 | `18_sample_efficiency` | Learning curves — shots needed before each architecture saturates |
| 19 | `19_generic_circuit_handler` | Auto-detects structure of any Robin-provided circuit; ready for d=7/d=9 |
| 20 | `20_heavyhex_grand_comparison` | Final comparison figures for all HeavyHex results |

---

## Key technical details

**Robin's LSTM bug fix (NB14–20):** The LSTM in NB01–13 used `det.reshape(-1, 9, 8)` — valid only for uniform circuits. The HeavyHex circuit has structure 12 initial + 9×24 + 12 final = 240 detectors. The correct temporal input is `(N, 11, 24)`, built explicitly via `build_temporal_input()`.

**Pickle version fix (NB14, 16, 17, 19):** Robin's circuits were saved with a different stim version, causing `ImportError: CompiledDemSampler already registered`. All affected notebooks use `load_circuit_from_pickle()` which extracts the ASCII circuit text directly from the raw pickle bytes.

**MWPM flag:** `approximate_disjoint_errors=True` is required for Robin's circuits because the ACES noise model produces non-decomposable correlated error mechanisms.

---

## Setup

```bash
pip install stim pymatching torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scikit-learn scipy
```

GPU strongly recommended. Total runtime for all notebooks: approximately 3 hours.

Place Robin's circuit pickles (`s_stim_circ_z.pickle`, `s_stim_circ_x.pickle`, `stim_circ_z.pickle`, `stim_circ_x.pickle`) in the project root before running notebooks 14–20.

---

## References

- Bausch et al. (AlphaQubit), *Learning to decode the surface code with a recurrent, transformer-based neural network*, Nature 635, 834 (2024) — arXiv:2310.05900
- Bödeker, Kusters & Müller, *On the interpretability of neural network decoders*, arXiv:2502.20269 (2025)
- Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021)
- Higgott & Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight matching*, arXiv:2303.15933

---

## Acknowledgements

Circuit generation and ACES noise model data provided by Robin Harper (University of Sydney). Built during a research internship with the Quantum Theory Group, School of Physics, University of Sydney.
