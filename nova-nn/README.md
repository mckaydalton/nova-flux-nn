
# NOvA NN — ND→FD Spectrum Prediction

Full project scaffold for training a Keras model that learns softmax mixture weights
over Near Detector (ND) sections to predict Far Detector (FD) spectra. Includes:

- `CombineNDandOsc` layer (fully serializable; works with `model.save()`)
- Config-driven training & evaluation (`configs/*.yaml`)
- Reproducible CLI scripts (`src/nova_nn/train.py`, `src/nova_nn/eval.py`)
- Tests, CI, pre-commit, and packaging

## Quickstart

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

# Run a tiny synthetic smoke test
python -m nova_nn.train --config configs/default.yaml

# Evaluate the saved model
python -m nova_nn.eval --model_path models/NOvA_NN.keras
```

See `docs/USAGE.md` for details on expected ND spectra formats and plugging in real data.
