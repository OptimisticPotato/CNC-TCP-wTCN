"""CNC -> TCP TCN.

Module map
----------
runs_io    reading ``Runs/<op>/*_out.csv`` (float64, dt assert, NaN mask)
resample   cubic-spline transport between arbitrary timestamps and the grid
features   increments, 6-channel G00/G01 masking, normalisation, loss weights
dataset    cache, program-level split, random crops, augmentation
model      causal bias-free dilated TCN
losses     MSE + diff-MSE + multi-resolution STFT, masked
trainer    training / validation loop, checkpoints
metrics    SPEC section 8 metrics
plots      SPEC section 8 figures
"""

__all__ = [
    "runs_io",
    "resample",
    "features",
    "dataset",
    "model",
    "losses",
    "trainer",
    "metrics",
    "plots",
]
