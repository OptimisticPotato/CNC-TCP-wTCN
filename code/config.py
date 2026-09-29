"""Single place to configure everything.

Expected project layout (paths resolve automatically for both of these):

    project/                      D:/Runs/              <- current layout
      Runs/                         DOE_0001_.../
        DOE_0001_.../               DOE_0002_.../
        DOE_0002_.../               code/
      code/                           config.py
        config.py

Override without editing this file:

  * per-machine       : create ``config_local.py`` next to this file (imported last)
  * environment       : TCN_RUNS_DIR, TCN_WORK_DIR
  * command line      : --set train.epochs=40 --set model.channels=64

Nothing else in the codebase hard-codes a path or a hyper-parameter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Tuple

CODE_DIR = Path(__file__).resolve().parent


def _default_runs_dir() -> Path:
    """Find the folder that holds the per-operation run folders."""
    env = os.environ.get("TCN_RUNS_DIR")
    if env:
        return Path(env)
    parent = CODE_DIR.parent
    sibling = parent / "Runs"                       # project/{Runs,code}
    if sibling.is_dir():
        return sibling
    if any(parent.glob("DOE_*")):                   # code/ lives inside Runs/
        return parent
    return sibling                                  # not there yet; fail loudly later


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    #: folder whose sub-folders are one machining operation each, every one
    #: containing a single ``*_out.csv`` from the simulator
    runs_dir: Path = field(default_factory=_default_runs_dir)

    #: everything this project writes lands here
    work_dir: Path = Path(os.environ.get("TCN_WORK_DIR", str(CODE_DIR / "work")))

    #: run folders listed here are never loaded (see the file for why)
    exclude_file: Path = CODE_DIR / "excluded_runs.txt"

    #: glob that recognises a run folder inside runs_dir
    run_glob: str = "DOE_*"

    #: glob that finds the simulator output inside a run folder
    csv_glob: str = "*_out.csv"

    @property
    def meta_dir(self) -> Path:      # run index + DOE selections
        return self.work_dir / "meta"

    @property
    def cache_dir(self) -> Path:     # per-run .npz training cache
        return self.work_dir / "cache"

    @property
    def runs_out_dir(self) -> Path:  # checkpoints / logs, one folder per training run
        return self.work_dir / "runs"

    @property
    def report_dir(self) -> Path:    # metric tables + figures
        return self.work_dir / "reports"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Data:
    #: standard grid the model always runs on [s]. SPEC section 6.2.
    #: This dataset shows servo resonances near 40 Hz AND near 116 Hz, so the
    #: "10x the dominant vibration" rule needs >= 1160 Hz: 1 ms is NOT enough
    #: here even though the spec allows it in general. Keep 0.5 ms.
    dt: float = 0.0005

    #: a run is accepted as already-uniform when
    #: max|diff(t) - dt| <= dt_tol * dt ; otherwise it is spline-resampled
    dt_tol: float = 1e-3

    #: runs whose native dt differs from ``dt`` by more than this factor are
    #: skipped rather than resampled (they cannot resolve the 116 Hz mode)
    dt_max_ratio: float = 5.4          # 2.7 ms / 0.5 ms, SPEC section 6.4

    #: rapid traverse feed [mm/min]. Rows with feed >= tol * rapid are G00,
    #: and it also normalises the input increments (u = dCNC / (rapid/60 * dt)).
    #:
    #: With ``rapid_feed_auto`` the value is measured from the dataset instead
    #: of trusted as a constant: 01_scan_runs writes the most common per-run
    #: maximum feed to work/meta/machine.json and everything reads it from
    #: there. The number below is the fallback when that file is missing, and
    #: the override when auto is off -- set it explicitly for a machine whose
    #: rapid is not what the data suggests.
    rapid_feed_mm_min: float = 40000.0
    rapid_feed_auto: bool = True
    rapid_feed_tol: float = 0.99       # G00 <=> feed >= tol * rapid
    #: a candidate rapid is only believed when it is this many times larger
    #: than the next distinct feed level in the dataset
    rapid_feed_min_gap: float = 1.5

    #: "feed"  -> G00/G01 from the feed column (exact; what the spec asks for)
    #: "speed" -> per-block commanded-speed threshold (fallback, ~99 % of blocks)
    #: "none"  -> u[3:] stays zero, i.e. 3 real input channels
    g00_source: str = "feed"
    g00_speed_threshold: float = 20000.0     # [mm/min], used by "speed"

    #: rows with empty cnc_*/tcp_* are never used as targets (SPEC section 3.3)
    drop_nan_targets: bool = True

    #: hold-out split, by PROGRAM (never by window). SPEC section 7.
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 20260922


# --------------------------------------------------------------------------- #
# DOE subset selection  (SPEC section 3.3 -- do not train on all ~1000)
# --------------------------------------------------------------------------- #
@dataclass
class Select:
    n_programs: int = 32              # first round; grow until the error saturates
    tag: str = "r1"                   # -> work/meta/selection_r1.txt
    exclude_substrings: Tuple[str, ...] = ()
    min_per_feed_bucket: int = 2      # stratification floors
    min_per_family: int = 1
    selection_seed: int = 7


# --------------------------------------------------------------------------- #
# model  (SPEC section 5)
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    in_channels: int = 6          # dCNC xyz * {G01, G00}
    out_channels: int = 3         # e = TCP - CNC  [um], X/Y/Z at once
    channels: int = 48            # spec says 32..64
    kernel_size: int = 5
    #: RF = 1 + (k-1) * sum(dilations)
    #:   k=5, (1..32)  ->  253 samples = 126 ms
    #:   k=5, (1..64)  ->  509 samples = 254 ms
    #:   k=5, (1..128) -> 1021 samples = 510 ms
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    activation: str = "relu"      # must satisfy f(0)=0 -> relu | tanh
    norm: str = "none"            # "none" | "layernorm".  BatchNorm is forbidden.
    dropout: float = 0.0
    #: bias-free linear causal conv straight from input to output (SPEC 5.2).
    #: Off by default; this is the first knob to turn if training stalls.
    linear_skip: bool = False
    linear_skip_taps: int = 0     # 0 -> use the full receptive field


# --------------------------------------------------------------------------- #
# loss  (SPEC sections 5.3 / 5.4)
# --------------------------------------------------------------------------- #
@dataclass
class Loss:
    lambda_diff: float = 0.5                  # weight of MSE(diff(e))
    lambda_stft: float = 0.3                  # weight of multi-resolution STFT
    stft_ffts: Tuple[int, ...] = (256, 512, 1024)
    stft_hops: Tuple[int, ...] = (64, 128, 256)
    g00_weight: float = 0.1                   # loss weight inside G00 blocks


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
@dataclass
class Train:
    window: int = 6144            # samples per crop (3.07 s at 0.5 ms)
    batch_size: int = 16          # 16 x 6144 x 48ch is small for 12 GB
    steps_per_epoch: int = 300
    epochs: int = 60
    lr: float = 2e-3
    weight_decay: float = 0.0     # no bias / no norm -> plain Adam is fine
    grad_clip: float = 1.0
    amp: bool = True              # mixed precision (Ampere tensor cores)
    num_workers: int = 0          # Windows: 0 is fastest, data is already in RAM
    seed: int = 0
    patience: int = 15            # early stop after N epochs without val gain (0=off)
    name: str = "tcn_r1"          # results go to work/runs/<name>/
    #: which validation number picks best.pt. The default is the G01 (cutting)
    #: RMSE, because that is where form error comes from; "val_rmse_um" scores
    #: every sample instead, and is then dominated by the rapid-traverse blocks
    #: -- a run can improve on that number while getting worse at the job.
    select_metric: str = "val_rmse_g01_um"


# --------------------------------------------------------------------------- #
# resample augmentation  (SPEC section 7 step 3)
# --------------------------------------------------------------------------- #
@dataclass
class Augment:
    enabled: bool = True
    prob: float = 0.5                              # chance a crop gets corrupted
    jitter: Tuple[float, ...] = (0.0, 0.25, 0.5)   # of the source dt
    decimate: Tuple[int, ...] = (1, 2, 4, 8)       # 8 -> a 4 ms source grid
    dropout_frac: Tuple[float, ...] = (0.0, 0.05)
    interp: str = "cubic"                          # PCHIP is forbidden (SPEC 6.4)


# --------------------------------------------------------------------------- #
# evaluation  (SPEC section 8)
# --------------------------------------------------------------------------- #
@dataclass
class Eval:
    rf_sweep_ms: Tuple[float, ...] = (100.0, 200.0, 300.0)   # metric 7
    ndoe_sweep: Tuple[int, ...] = (8, 16, 32, 64)            # metric 6
    export_tcp_csv: bool = True   # hand-off to an external form-error pipeline
    nperseg: int = 2048           # PSD / coherence window (metric 4)
    #: taps for the FIR reference (metric 9). Measured on this dataset with
    #: clean programs and plain least squares, held-out G01 RMSE:
    #:   200 taps (100 ms) 3.2 um | 509 (254 ms) 0.18 um | 1021 (510 ms) 0.005 um
    #: The truncation tail decays with tau ~ 50-70 ms, so 100 ms is far too
    #: short here even though the dominant 37 Hz mode settles in ~12 ms.
    fir_taps: int = 1021
    psd_band_hz: Tuple[float, float] = (0.0, 200.0)


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: Data = field(default_factory=Data)
    select: Select = field(default_factory=Select)
    model: Model = field(default_factory=Model)
    loss: Loss = field(default_factory=Loss)
    train: Train = field(default_factory=Train)
    augment: Augment = field(default_factory=Augment)
    eval: Eval = field(default_factory=Eval)

    # ---------------- derived ----------------
    @property
    def receptive_field(self) -> int:
        """Past samples the TCN can see."""
        return 1 + (self.model.kernel_size - 1) * sum(self.model.dilations)

    @property
    def receptive_field_ms(self) -> float:
        return self.receptive_field * self.data.dt * 1e3

    @property
    def increment_scale(self) -> float:
        """mm travelled in one sample at rapid feed -- normalises the input.

        Deliberately the *configured* constant, never the auto-detected rapid:
        a trained network expects the input scaling it was trained with, so
        this must not move when the data changes. Checkpoints record it and
        the loader restores it.
        """
        return self.data.rapid_feed_mm_min / 60.0 * self.data.dt

    def describe(self) -> str:
        return (
            f"runs dir        : {self.paths.runs_dir}\n"
            f"work dir        : {self.paths.work_dir}\n"
            f"standard grid   : {self.data.dt * 1e3:.3f} ms "
            f"({1.0 / self.data.dt:.0f} Hz)\n"
            f"receptive field : {self.receptive_field} samples "
            f"({self.receptive_field_ms:.1f} ms)\n"
            f"increment scale : {self.increment_scale:.6f} mm/sample"
        )


CFG = Config()


# --------------------------------------------------------------------------- #
# overrides
# --------------------------------------------------------------------------- #
def _coerce(old: Any, text: str) -> Any:
    if isinstance(old, bool):
        return text.lower() in ("1", "true", "yes", "on")
    if isinstance(old, Path):
        return Path(text)
    if isinstance(old, tuple):
        inner = type(old[0]) if old else float
        return tuple(inner(x) for x in text.replace(" ", "").split(",") if x != "")
    if isinstance(old, int):
        return int(text)
    if isinstance(old, float):
        return float(text)
    return text


def apply_override(cfg: Config, spec: str) -> None:
    """apply_override(CFG, "train.epochs=40")"""
    key, _, value = spec.partition("=")
    node: Any = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = getattr(node, p)
    leaf = parts[-1]
    if not hasattr(node, leaf):
        raise KeyError(f"unknown config key: {key}")
    setattr(node, leaf, _coerce(getattr(node, leaf), value.strip()))


def add_config_args(parser) -> None:
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override any config value, e.g. --set train.epochs=40",
    )


def configure(args=None) -> Config:
    for spec in getattr(args, "overrides", None) or []:
        apply_override(CFG, spec)
    for d in (CFG.paths.work_dir, CFG.paths.meta_dir, CFG.paths.cache_dir,
              CFG.paths.runs_out_dir, CFG.paths.report_dir):
        d.mkdir(parents=True, exist_ok=True)
    return CFG


def as_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj


# per-machine overrides, imported last so they win
try:  # pragma: no cover
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
