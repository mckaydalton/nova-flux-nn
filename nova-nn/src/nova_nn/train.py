
import argparse
import os
import yaml
import numpy as np
import tensorflow as tf

from .data import load_nd_spectra
from .model import build_model

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    # Data
    dcfg = cfg["data"]
    n_sections = dcfg["n_sections"]
    n_bins = dcfg["n_bins"]
    d_embed = dcfg["d_embed"]

    nd = load_nd_spectra(dcfg.get("nd_spectra_path"), n_sections, n_bins)

    # Synthetic training data (unless user provides FD targets in the future)
    steps = dcfg.get("steps_per_epoch", 32)
    batch = dcfg.get("batch_size", 32)
    x_train = np.random.randn(steps * batch, d_embed).astype("float32")
    # Create ground truth via a hidden projection to ND weights, then combine
    # (this gives the model a learnable target consistent with the architecture)
    hidden = np.random.randn(d_embed, n_sections).astype("float32")
    logits = x_train @ hidden
    weights = tf.nn.softmax(logits, axis=-1).numpy()
    y_train = weights @ nd

    # Model
    mcfg = cfg["model"]
    model = build_model(n_sections, n_bins, nd, d_embed= d_embed,
                        temperature=mcfg.get("temperature", 1.0),
                        dropout_rate=mcfg.get("dropout_rate", 0.1))

    # Train
    tcfg = cfg["train"]
    lr = tcfg.get("lr", 1e-3)
    epochs = tcfg.get("epochs", 5)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=tcfg.get("early_stopping", {}).get("patience", 5),
        min_delta=tcfg.get("early_stopping", {}).get("min_delta", 1e-4),
        restore_best_weights=True
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch,
        callbacks=[es],
        verbose=1
    )

    # Save
    out = cfg["out"]
    os.makedirs(out["model_dir"], exist_ok=True)
    model_path = os.path.join(out["model_dir"], out["model_name"])
    model.save(model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
