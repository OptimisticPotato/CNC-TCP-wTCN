"""Cache, program-level split, random crops, resample augmentation.

SPEC section 7: *never* split by window. Neighbouring windows overlap, so a
random window split leaks the answer into validation and the reported error
becomes fiction. Programs go into exactly one of train / val / test.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import features, resample
from .runs_io import family_of, load_run


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def cache_path(cfg, name: str) -> Path:
    return Path(cfg.paths.cache_dir) / f"{name}.npz"


def cache_stamp(cfg) -> np.ndarray:
    """Everything the cached arrays depend on.

    A cache entry that was built under a different standard grid or a
    different G00 rule is not a slow cache, it is wrong data. The stamp is
    written next to the arrays and checked on load, so changing the config
    rebuilds instead of silently reusing.
    """
    from .runs_io import rapid_feed
    return np.array([
        float(cfg.data.dt),
        float(rapid_feed(cfg)),
        float(cfg.data.rapid_feed_tol),
        # zlib.crc32, not hash(): str hashing is salted per process, which
        # would make every run see its own cache as stale
        float(zlib.crc32(str(cfg.data.g00_source).encode())),
        float(cfg.data.g00_speed_threshold),
        float(bool(cfg.data.drop_nan_targets)),
    ], dtype=np.float64)


def build_cache(cfg, names: Sequence[str], force: bool = False,
                verbose: bool = True) -> List[str]:
    """Parse the CSVs once and keep float64 arrays on disk. ~1.5 MB per run."""
    Path(cfg.paths.cache_dir).mkdir(parents=True, exist_ok=True)
    stamp = cache_stamp(cfg)
    done, rebuilt = [], 0
    for i, name in enumerate(names, 1):
        p = cache_path(cfg, name)
        if p.exists() and not force and _stamp_matches(p, stamp):
            done.append(name)
            continue
        if p.exists() and not force:
            rebuilt += 1
        run = load_run(cfg, name)
        g00 = features.g00_mask(run, cfg)
        np.savez(
            p,
            cnc=run.cnc, tcp=run.tcp, block=run.block.astype(np.int32),
            valid=run.valid, g00=g00, feed=run.feed.astype(np.float32),
            t0=np.float64(run.t[0]), dt=np.float64(run.dt),
            dt_native=np.float64(run.dt_native), resampled=np.bool_(run.resampled),
            stamp=stamp,
        )
        done.append(name)
        if verbose:
            print(f"  [{i}/{len(names)}] cached {name}  n={len(run)}  "
                  f"g00={g00.mean():.2f}", flush=True)
    if rebuilt:
        print(f"  rebuilt {rebuilt} stale cache entries "
              f"(dt / G00 rule changed since they were written)")
    return done


def _stamp_matches(path: Path, stamp: np.ndarray) -> bool:
    try:
        with np.load(path) as z:
            return "stamp" in z and np.array_equal(z["stamp"], stamp)
    except (OSError, ValueError, EOFError):
        return False


@dataclass
class Program:
    name: str
    cnc: np.ndarray        # (N,3) float64 mm
    e: np.ndarray          # (N,3) float32 um   (target)
    u: np.ndarray          # (N,6) float32      (input, pre-computed)
    g00: np.ndarray        # (N,)  bool
    valid: np.ndarray      # (N,)  bool
    block: np.ndarray      # (N,)  int32
    dt: float

    def __len__(self) -> int:
        return self.cnc.shape[0]

    @property
    def family(self) -> str:
        return family_of(self.name)


def load_program(cfg, name: str) -> Program:
    p = cache_path(cfg, name)
    if not p.exists() or not _stamp_matches(p, cache_stamp(cfg)):
        build_cache(cfg, [name], verbose=False)
    z = np.load(p)
    cnc = z["cnc"].astype(np.float64)
    tcp = z["tcp"].astype(np.float64)
    g00 = z["g00"].astype(bool)
    u = features.build_inputs(cnc, g00, cfg.increment_scale)
    features.check_identity(u, cnc, cfg.increment_scale)
    return Program(
        name=name, cnc=cnc, e=features.build_target(cnc, tcp), u=u, g00=g00,
        valid=z["valid"].astype(bool), block=z["block"].astype(np.int32),
        dt=float(z["dt"]),
    )


def load_programs(cfg, names: Sequence[str], verbose: bool = True) -> List[Program]:
    out = []
    for n in names:
        out.append(load_program(cfg, n))
        if verbose:
            print(f"  loaded {n}  n={len(out[-1])}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def split_programs(cfg, names: Sequence[str]) -> Dict[str, List[str]]:
    """Deterministic program-level split. Families are spread across the three
    sets so that val/test are not accidentally made of one path type."""
    rng = np.random.default_rng(cfg.data.split_seed)
    by_family: Dict[str, List[str]] = {}
    for n in sorted(names):
        by_family.setdefault(family_of(n), []).append(n)

    n_total = len(names)
    n_val = max(1, int(round(cfg.data.val_fraction * n_total)))
    n_test = max(1, int(round(cfg.data.test_fraction * n_total)))

    for fam in sorted(by_family):
        rng.shuffle(by_family[fam])

    # round-robin over families keeps each split diverse
    ordered: List[str] = []
    fams = sorted(by_family)
    idx = {f: 0 for f in fams}
    while len(ordered) < n_total:
        for f in fams:
            if idx[f] < len(by_family[f]):
                ordered.append(by_family[f][idx[f]])
                idx[f] += 1

    test = ordered[:n_test]
    val = ordered[n_test:n_test + n_val]
    train = ordered[n_test + n_val:]
    if not train:
        raise ValueError("not enough programs to build a train split")
    # train keeps the family round-robin order, so that a prefix of it is still
    # a diverse subset -- that is what the n-programs sweep slices
    return {"train": train, "val": sorted(val), "test": sorted(test)}


def save_split(path: Path, split: Dict[str, List[str]]) -> None:
    Path(path).write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split(path: Path) -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Random crops from the training programs.

    Every crop is treated as a standalone causal sequence: its first
    ``warmup`` (= receptive field) outputs have no history and are therefore
    excluded from the loss, exactly as SPEC section 5.4 requires.
    """

    def __init__(self, cfg, programs: Sequence[Program], augment: bool = True,
                 length: Optional[int] = None, epoch_size: Optional[int] = None,
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.programs = list(programs)
        self.augment = augment and cfg.augment.enabled
        self.window = int(length or cfg.train.window)
        self.warmup = cfg.receptive_field
        self.epoch_size = int(epoch_size or cfg.train.steps_per_epoch * cfg.train.batch_size)
        self.seed = cfg.train.seed if seed is None else seed
        if self.window <= self.warmup:
            raise ValueError(
                f"train.window ({self.window}) must exceed the receptive field "
                f"({self.warmup}); nothing would be left to score"
            )
        w = np.array([max(1, int(p.valid.sum())) for p in self.programs], dtype=np.float64)
        self.p_sel = w / w.sum()

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int):
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2 ** 63))
        prog = self.programs[int(rng.choice(len(self.programs), p=self.p_sel))]
        n = len(prog)
        w = min(self.window, n)
        start = int(rng.integers(0, n - w + 1)) if n > w else 0
        sl = slice(start, start + w)

        cnc = prog.cnc[sl]
        g00 = prog.g00[sl]
        e = prog.e[sl]
        valid = prog.valid[sl]

        if self.augment:
            aug = resample.sample_augmentation(self.cfg, rng)
            if aug["active"]:
                cnc = resample.corrupt_and_restore(
                    cnc, prog.dt, rng, jitter=aug["jitter"],
                    decimate=aug["decimate"], drop_frac=aug["drop_frac"],
                    kind=self.cfg.augment.interp,
                )
            u = features.build_inputs(cnc, g00, self.cfg.increment_scale)
        else:
            u = prog.u[sl]

        weight = features.loss_weight(valid, g00, self.warmup, self.cfg.loss.g00_weight)

        if w < self.window:                      # zero-pad short programs
            pad = self.window - w
            u = np.pad(u, ((0, pad), (0, 0)))
            e = np.pad(e, ((0, pad), (0, 0)))
            weight = np.pad(weight, (0, pad))

        return (
            torch.from_numpy(np.ascontiguousarray(u.T)),        # (6, W)
            torch.from_numpy(np.ascontiguousarray(e.T)),        # (3, W)
            torch.from_numpy(np.ascontiguousarray(weight)),     # (W,)
        )


def make_loader(cfg, programs: Sequence[Program], augment: bool,
                shuffle: bool = False, seed: Optional[int] = None) -> DataLoader:
    ds = WindowDataset(cfg, programs, augment=augment, seed=seed)
    # persistent_workers is deliberately off: the trainer bumps
    # ``dataset.seed`` every epoch to get fresh crops, and a persistent worker
    # would keep its own copy of the old value and replay the same windows.
    return DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
        num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, persistent_workers=False,
    )


def full_sequence(cfg, prog: Program) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole program as one causal sequence, for validation / evaluation."""
    weight = features.loss_weight(prog.valid, prog.g00, cfg.receptive_field,
                                  cfg.loss.g00_weight)
    return (
        torch.from_numpy(np.ascontiguousarray(prog.u.T))[None],
        torch.from_numpy(np.ascontiguousarray(prog.e.T))[None],
        torch.from_numpy(np.ascontiguousarray(weight))[None],
    )


@torch.no_grad()
def predict_program(model, cfg, prog: Program, device, chunk: int = 200_000,
                    amp: bool = False) -> np.ndarray:
    """Predicted e for a whole program, (N,3) float32 [um].

    Long programs are processed in chunks that carry ``receptive_field`` samples
    of overlap, so the result is identical to a single pass.
    """
    model.eval()
    n = len(prog)
    rf = cfg.receptive_field
    out = np.zeros((n, 3), dtype=np.float32)
    u_all = torch.from_numpy(np.ascontiguousarray(prog.u.T))[None]
    start = 0
    while start < n:
        stop = min(n, start + chunk)
        lo = max(0, start - rf)
        x = u_all[..., lo:stop].to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            y = model(x)
        y = y.float()[0].T.cpu().numpy()
        out[start:stop] = y[start - lo:]
        start = stop
    return out
