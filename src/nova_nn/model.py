
import tensorflow as tf
from .layers import CombineNDandOsc

def build_model(n_sections, n_bins, nd_spectra, d_embed=64, temperature=1.0, dropout_rate=0.1):
    inp = tf.keras.Input(shape=(d_embed,))
    z = tf.keras.layers.Dense(128, activation="relu")(inp)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(64, activation="relu")(z)
    fd = CombineNDandOsc(
        nd_spectra=nd_spectra,
        temperature=temperature,
        dropout_rate=dropout_rate,
        name="combine_nd_osc"
    )(z)
    model = tf.keras.Model(inp, fd, name="nova_nn")
    return model
