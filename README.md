# Neural Network Decoders for Quantum Error Correction

Comparing neural network decoders (MLP, LSTM) against classical Minimum Weight Perfect Matching (MWPM) for a distance-3 rotated surface code simulated with [Stim](https://github.com/quantumlib/Stim).

---

## Results

| Decoder | LER at p=0.001 | vs Trivial |
|---|---|---|
| Trivial (always predict 0) | 5.47% | — |
| MLP — raw syndromes (Part 2) | 3.12% | 1.8× |
| MLP — detection events (Part 1) | 1.34% | 4.1× |
| LSTM — detection events | TBD | TBD |
| MWPM (PyMatching) | TBD | TBD |

![LER vs noise](figures/fig1_ler_vs_noise_loglog.png)

---

## Setup

```bash
pip install stim pymatching torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scikit-learn
```

---

## Notebooks

Run in order:

| Notebook | What it does |
|---|---|
| `01_data_generation.ipynb` | Builds Stim circuit, samples 100k shots, generates sweep data |
| `02_decoder_training.ipynb` | Trains MLP decoders (Part 1 and Part 2) |
| `03_noise_sweep.ipynb` | Evaluates MLP decoders across noise levels |
| `04_pymatching_benchmark.ipynb` | Runs MWPM via PyMatching, generates reference comparison |
| `05_lstm_decoder.ipynb` | Trains LSTM decoder with temporal syndrome structure |
| `06_comparison_figures.ipynb` | Produces all 6 presentation figures |

---

## Folder Structure

```
project/
├── data/
│   ├── detection_events.npy     # (100k, 72) — Part 1 input
│   ├── raw_measurements.npy     # (100k, 72) — Part 2 input (syndrome cols only)
│   ├── observable_flips.npy     # (100k, 1)  — shared labels
│   └── sweep/                   # 50k shots × 6 noise levels
├── models/
│   ├── decoder_part1.pt         # MLP trained on detection events
│   ├── decoder_part2.pt         # MLP trained on raw syndromes
│   └── decoder_lstm.pt          # LSTM decoder
├── results/
│   ├── mwpm_sweep.npy
│   └── lstm_sweep.npy
├── figures/                     # All 6 presentation figures
├── 01_data_generation.ipynb
├── 02_decoder_training.ipynb
├── 03_noise_sweep.ipynb
├── 04_pymatching_benchmark.ipynb
├── 05_lstm_decoder.ipynb
└── 06_comparison_figures.ipynb
```

---

## Key Design Decisions

**Single sampling run:** Both Part 1 and Part 2 use detection events and raw measurements from the same Stim sampling run, ensuring they share identical `observable_flips` labels — enabling a fair comparison.

**Syndrome-only columns:** The raw measurement array is sliced to columns 0–71 (syndrome qubits) only. Columns 72–80 are the final data qubit readout, which XOR directly to the observable — including them makes Part 2 trivially solvable.

**LSTM temporal structure:** The LSTM reshapes input from `(N, 72)` flat to `(N, 9 rounds, 8 stabilisers)`, allowing the network to process syndrome measurements as a time series and carry hidden state across rounds.

---

## Reference

Bödeker et al., *On the interpretability of neural network decoders*, arXiv:2502.20269 (2025)
