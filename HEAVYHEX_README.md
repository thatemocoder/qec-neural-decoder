# HeavyHex Circuit Series — README

## What these notebooks are

Notebooks 14–20 work with circuits provided by Robin, which are fundamentally
different from the simple d=3 rotated surface code used in notebooks 01–13.

## The circuits

| File | Code | Distance | Basis | Rounds | Detectors |
|---|---|---|---|---|---|
| `s_stim_circ_z.pickle` | Standard HeavyHex | 5 | Z | 9 | 240 |
| `s_stim_circ_x.pickle` | Standard HeavyHex | 5 | X | 9 | 240 |
| `stim_circ_z.pickle` | Dynamic HeavyHex | 5 | Z | 9 | 236 |
| `stim_circ_x.pickle` | Dynamic HeavyHex | 5 | X | 9 | 236 |

**Start with `s_stim_circ_z` (Standard Z basis)** — Robin recommends this.

**Coming from Robin:** circuits for d=7 and d=9 with richer noise models.
When they arrive, add them to the `CIRCUITS` dict in notebook 19 and re-run.

## Why these circuits are more realistic

- **65 physical qubits** on IBM Heron r2/r3 Heavy-Hex topology
- **Real ACES noise model** — calibrated from actual ibm_boston device data.
  Every qubit and gate has its own measured error rate (not a single depolarising p)
- **HeavyHex connectivity** — the physical qubit graph used by IBM quantum hardware,
  not the idealised nearest-neighbour grid

## The critical structural difference (Robin's bug report)

Our previous LSTM (notebook 05) used:
```python
det.reshape(-1, 9, 8)   # WRONG for this circuit
```
That worked for the simple d=3 circuit where all 9 rounds were identical.

For `s_stim_circ_z` the 240 detectors have a non-uniform structure:

```
Block 0 (initial):     12 detectors  ← prep round
Blocks 1-9 (9 rounds): 24 detectors each
    First  12: Z-type stabilisers (XOR with previous round)
    Second 12: X-type stabilisers (simultaneous checks)
Block 10 (final):      12 detectors  ← data qubit readout
```

The correct temporal input for LSTM/Transformer is `(N, 11, 24)`:
```python
seq[:, 0,    :12] = det[:, 0:12]              # initial, padded
seq[:, 1:10, :]   = det[:, 12:228].reshape(N, 9, 24)  # 9 rounds
seq[:, 10,   :12] = det[:, 228:240]           # final, padded
```

This is implemented in `build_temporal_input()` in notebook 14.

## MWPM flag

Always use `approximate_disjoint_errors=True` when building the DEM:
```python
dem = circuit.detector_error_model(
    decompose_errors=True,
    approximate_disjoint_errors=True   # required for ACES noise
)
```

## Run order

```
14 → 15 → 16 → 17 → 18 → 19 → 20
```

| Notebook | Time | Purpose |
|---|---|---|
| 14 | ~2 min | Load circuit, sample data, MWPM baseline |
| 15 | ~15 min | Train MLP, LSTM-det, LSTM-raw, Transformer |
| 16 | ~5 min | Decay curve (LER vs rounds) |
| 17 | ~3 min | Speed benchmark |
| 18 | ~45 min | Learning curves (sample efficiency) |
| 19 | ~20 min | Generic handler (add d=7/d=9 here when Robin provides them) |
| 20 | ~2 min | Grand comparison figures |

## Folder structure created

```
data/
├── heavyhex/                    ← created by NB14
│   ├── detection_events.npy    (N, 240)   flat for MLP
│   ├── det_temporal.npy        (N, 11, 24) for LSTM/Transformer
│   ├── raw_measurements.npy    (N, 425)   for LSTM-raw
│   └── observable_flips.npy   (N,)
├── heavyhex_s_stim_circ_z/     ← created by NB19
│   ├── structure.json
│   └── ...
models/
├── heavyhex/                   ← created by NB15
│   ├── mlp.pt
│   ├── lstm_det.pt
│   ├── lstm_raw.pt
│   └── transformer.pt
results/
├── heavyhex/
│   └── mwpm_result.npy
figures/
├── 15_heavyhex_training.png
├── 16_decay_curve.png
├── 17_speed_vs_accuracy.png
├── 18_learning_curve.png
├── fig_hh_decoder_comparison.png
└── fig_hh_raw_vs_det.png
```

## Open tasks waiting on Robin

1. **d=7 circuit** — `s_stim_circ_z_d7.pickle` + `s_stim_circ_x_d7.pickle`
2. **d=9 circuit with richer noise model** — `s_stim_circ_z_d9.pickle`
3. **Circuits at different round counts** for decay curve:
   `s_stim_circ_z_r1.pickle`, `_r2`, `_r4`, `_r8`, `_r12`, `_r16`, `_r20`

When any of these arrive: add the file to the `CIRCUITS` dict in notebook 19
and set `ROUNDS_TO_DO` in notebook 16. Everything else adapts automatically.
