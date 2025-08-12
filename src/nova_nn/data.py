
import numpy as np
import pandas as pd
from typing import Optional, Tuple

def load_nd_spectra(path: Optional[str], n_sections: int, n_bins: int) -> np.ndarray:
    if path is None:
        # synthetic positive ND spectra
        nd = np.abs(np.random.randn(n_sections, n_bins)).astype("float32")
        return nd
    if path.endswith(".npy"):
        arr = np.load(path)
        assert arr.ndim == 2, "ND spectra must be 2D (n_sections, n_bins)"
        return arr.astype("float32")
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        arr = df.values
        assert arr.ndim == 2, "ND spectra CSV must be 2D"
        return arr.astype("float32")
    else:
        raise ValueError("Unsupported ND spectra format. Use .npy or .csv")
