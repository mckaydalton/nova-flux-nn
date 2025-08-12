
import numpy as np
import tensorflow as tf
from nova_nn.layers import CombineNDandOsc

def test_serialize_roundtrip():
    n_sections, n_bins, d = 4, 10, 12
    nd = np.abs(np.random.randn(n_sections, n_bins)).astype("float32")

    inp = tf.keras.Input(shape=(d,))
    z = tf.keras.layers.Dense(8, activation="relu")(inp)
    out = CombineNDandOsc(nd_spectra=nd, temperature=1.0, dropout_rate=0.0)(z)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")

    x = np.random.randn(2, d).astype("float32")
    y = np.random.randn(2, n_bins).astype("float32")
    model.fit(x, y, epochs=1, verbose=0)

    model.save("/mnt/data/tmp_model.keras")
    loaded = tf.keras.models.load_model("/mnt/data/tmp_model.keras", compile=False)

    y1 = model(x, training=False).numpy()
    y2 = loaded(x, training=False).numpy()
    assert np.allclose(y1, y2, atol=1e-6)
