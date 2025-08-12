
import numpy as np
import tensorflow as tf
from nova_nn.model import build_model

def main():
    n_sections, n_bins, d = 8, 60, 64
    nd = np.abs(np.random.randn(n_sections, n_bins)).astype("float32")

    model = build_model(n_sections, n_bins, nd, d_embed=d)
    model.compile(optimizer="adam", loss="mse")

    x = np.random.randn(64, d).astype("float32")
    y = np.random.randn(64, n_bins).astype("float32")
    model.fit(x, y, epochs=1, verbose=1)

    model.save("models/NOvA_NN.keras")
    _ = tf.keras.models.load_model("models/NOvA_NN.keras", compile=False)
    print("Saved and reloaded OK.")

if __name__ == "__main__":
    main()
