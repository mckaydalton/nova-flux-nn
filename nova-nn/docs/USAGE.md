
# Usage

## Data formats

- **ND spectra**: 2D array `(n_sections, n_bins)`. Supported:
  - `.npy` (recommended)
  - `.csv` with header row; `n_bins` columns.
- **Targets (FD spectra)**: 2D array `(n_examples, n_bins)` or generated synthetically.

Update `configs/default.yaml` to point to your data files.

## Config

Key sections in YAML:
- `data`: paths and sharding
- `model`: layer hyperparameters
- `train`: optimizer, epochs, callbacks
- `out`: save paths

## CLI

```bash
python -m nova_nn.train --config configs/your.yaml
python -m nova_nn.eval --model_path models/NOvA_NN.keras
```

## Replacing synthetic data

Set `data.nd_spectra_path` to your ND `.npy`/`.csv`, and provide FD targets via
`data.fd_targets_path`. If not provided, the trainer makes synthetic targets to smoke-test.
