
import argparse
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, compile=False)
    # Simple smoke test forward pass
    input_shape = model.inputs[0].shape
    d = int(input_shape[-1])
    x = np.random.randn(2, d).astype("float32")
    y = model.predict(x, verbose=0)
    print("Inference OK. Output shape:", y.shape)

if __name__ == "__main__":
    main()
