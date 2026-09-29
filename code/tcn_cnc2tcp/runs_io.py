"""Reading the simulator output.

One run folder = one machining operation = one ``*_out.csv``:

    time,tcode,feed,spindlespeed,block#,cam_x,cam_y,cam_z,cnc_x,cnc_y,cnc_z,
    tcp_x,tcp_y,tcp_z,force_x,force_y,force_z,chipthickness
    sec,,mm/min,rpm,,mm,mm,mm,mm,mm,mm,N,N,N,mm
    0.0005,8,40000,10000,6,320, 320, 400,320, 320, 400,320, 320, 400,,,,,

Three things bite here and are handled once, in this module:

* **row 2 is a units row** -> skipped
* **data rows carry one field more than the header** (trailing comma). Letting
  pandas guess shifts every column by one and silently turns ``spindlespeed``
  into ``feed``. We pad the name list instead, which is exact.
* **empty cells** (tool change, end of file) -> NaN, never used as targets.

Coordinates are read as float64 and stay float64 (SPEC section 9: 320 mm +- um
does not survive float32).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

AXES = ("x", "y", "z")
CNC_COLS = tuple(f"cnc_{a}" for a in AXES)
TCP_COLS = tuple(f"tcp_{a}" for a in AXES)
REQUIRED = ("time", "feed", "block#") + CNC_COLS + TCP_COLS

#: spare column names appended to the header so that trailing extra fields
#: land in throw-away columns instead of shifting real ones
_N_PAD = 4


class RunError(RuntimeError):
    pass


@dataclass
class Run:
    """One operation on a uniform time grid."""

    name: str
    csv: Path
    t: np.ndarray            # (N,)   float64, seconds
    cnc: np.ndarray          # (N,3)  float64, mm
    tcp: np.ndarray          # (N,3)  float64, mm
    feed: np.ndarray         # (N,)   float64, mm/min (per-block modal value)
    block: np.ndarray        # (N,)   int64
    valid: np.ndarray        # (N,)   bool -- usable as a target
    dt: float
    dt_native: float
    resampled: bool

    def __len__(self) -> int:
        return self.t.size

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self) else 0.0

    @property
    def error_um(self) -> np.ndarray:
        """e = TCP - CNC in micrometres -- the quantity the model predicts."""
        return (self.tcp - self.cnc) * 1e3


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def read_exclude_file(path: Path) -> List[str]:
    if not path or not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def list_runs(cfg, include_excluded: bool = False) -> List[str]:
    """Names of every usable run folder under ``cfg.paths.runs_dir``."""
    root = Path(cfg.paths.runs_dir)
    if not root.is_dir():
        raise RunError(
            f"runs_dir does not exist: {root}\n"
            f"point at it with --set paths.runs_dir=... or TCN_RUNS_DIR"
        )
    excluded = set() if include_excluded else set(read_exclude_file(cfg.paths.exclude_file))
    names = []
    for d in sorted(root.glob(cfg.paths.run_glob)):
        if not d.is_dir() or d.name in excluded:
            continue
        if not any(d.glob(cfg.paths.csv_glob)):
            continue
        names.append(d.name)
    return names


def run_csv_path(cfg, name: str) -> Path:
    d = Path(cfg.paths.runs_dir) / name
    hits = sorted(d.glob(cfg.paths.csv_glob))
    if not hits:
        raise RunError(f"no {cfg.paths.csv_glob} inside {d}")
    if len(hits) > 1:
        raise RunError(f"{d} holds {len(hits)} csv files, expected 1: {[h.name for h in hits]}")
    return hits[0]


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def read_csv(path: Path, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Parse one ``*_out.csv`` into a float64 frame, columns by name.

    The data rows carry one field more than the header (a trailing comma). The
    name list is therefore padded to the widest row that actually occurs, so
    the extra fields land in throw-away columns instead of shifting every real
    column by one -- which is what happens if pandas is left to guess.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = [h.strip() for h in fh.readline().rstrip("\r\n").split(",")]
        fh.readline()                                     # units row
        widest = 0
        for i, line in enumerate(fh):
            widest = max(widest, line.count(",") + 1)
            if i >= 2000:
                break
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise RunError(f"{path.name}: missing column(s) {missing}; header={header}")

    want = list(REQUIRED) + [c for c in extra_cols if c in header]
    n_extra = max(0, widest - len(header))
    for extra in range(n_extra, n_extra + _N_PAD + 1):
        names = header + [f"__pad{i}" for i in range(extra)]
        try:
            return pd.read_csv(
                path, header=None, skiprows=2, names=names, usecols=want,
                encoding="utf-8-sig", na_values=["", " "], keep_default_na=True,
                dtype=np.float64,
            )
        except pd.errors.ParserError as exc:
            # only "a row further down is wider than anything we sampled" is
            # worth another attempt; everything else is a real problem and
            # must not be hidden behind four more tries
            if "Expected" not in str(exc) or "saw" not in str(exc):
                raise RunError(f"{path.name}: {exc}") from exc
    raise RunError(f"{path.name}: rows wider than header + {_N_PAD} fields")


def _uniform_dt(t: np.ndarray, dt_tol: float) -> tuple[float, bool]:
    if t.size < 3:
        return float("nan"), False
    d = np.diff(t)
    dt = float(np.median(d))
    uniform = bool(np.max(np.abs(d - dt)) <= dt_tol * dt)
    return dt, uniform


def load_run(cfg, name: str, allow_resample: bool = True) -> Run:
    """Read one run and put it on the standard grid.

    Uniform runs whose native dt already equals ``cfg.data.dt`` are used as-is.
    Anything else goes through the cubic-spline path of SPEC section 6.2.
    """
    csv = run_csv_path(cfg, name)
    df = read_csv(csv)

    t = df["time"].to_numpy(np.float64)
    cnc = df[list(CNC_COLS)].to_numpy(np.float64)
    tcp = df[list(TCP_COLS)].to_numpy(np.float64)
    feed = df["feed"].to_numpy(np.float64)
    block = df["block#"].to_numpy(np.float64)

    good_t = np.isfinite(t)
    if not good_t.all():                                  # never seen, but cheap
        t, cnc, tcp, feed, block = (a[good_t] for a in (t, cnc, tcp, feed, block))
    order = np.argsort(t, kind="stable")
    if not np.all(order == np.arange(t.size)):
        t, cnc, tcp, feed, block = (a[order] for a in (t, cnc, tcp, feed, block))

    dt_native, uniform = _uniform_dt(t, cfg.data.dt_tol)
    if not np.isfinite(dt_native) or dt_native <= 0:
        raise RunError(f"{name}: cannot determine dt")
    ratio = dt_native / cfg.data.dt
    if ratio > cfg.data.dt_max_ratio:
        raise RunError(
            f"{name}: native dt {dt_native * 1e3:.3f} ms is coarser than the "
            f"{cfg.data.dt * 1e3 * cfg.data.dt_max_ratio:.2f} ms limit "
            f"(SPEC 6.4: the standard grid must resolve the dominant vibration)"
        )

    on_grid = uniform and abs(dt_native - cfg.data.dt) <= cfg.data.dt_tol * cfg.data.dt
    resampled = False
    if not on_grid:
        if not allow_resample:
            raise RunError(f"{name}: dt={dt_native * 1e3:.4f} ms, uniform={uniform}")
        from .resample import run_to_grid
        t, cnc, tcp, feed, block = run_to_grid(
            t, cnc, tcp, feed, block, cfg.data.dt, kind=cfg.augment.interp
        )
        resampled = True

    valid = np.isfinite(cnc).all(1) & np.isfinite(tcp).all(1)
    if not cfg.data.drop_nan_targets:
        valid = np.ones(t.size, bool)

    block = np.where(np.isfinite(block), block, -1).astype(np.int64)
    feed = _fill_feed(feed)

    return Run(
        name=name, csv=csv, t=t, cnc=cnc, tcp=tcp, feed=feed, block=block,
        valid=valid, dt=float(cfg.data.dt), dt_native=float(dt_native),
        resampled=resampled,
    )


def _fill_feed(feed: np.ndarray) -> np.ndarray:
    """Forward/backward fill the modal feed across empty rows."""
    out = feed.copy()
    bad = ~np.isfinite(out)
    if bad.all():
        return np.zeros_like(out)
    if bad.any():
        idx = np.where(~bad)[0]
        out[bad] = np.interp(np.where(bad)[0], idx, out[idx])
    return out


# --------------------------------------------------------------------------- #
# quick statistics, used by the run index and the DOE selector
# --------------------------------------------------------------------------- #
def run_stats(cfg, name: str) -> Dict[str, object]:
    from .features import g00_mask

    run = load_run(cfg, name)
    g00 = g00_mask(run, cfg)
    d = np.diff(run.cnc, axis=0, prepend=run.cnc[:1])
    step = np.linalg.norm(d, axis=1)
    speed = step / run.dt * 60.0                            # mm/min
    e = run.error_um[run.valid]

    # per-block geometry
    blk = run.block
    edges = np.flatnonzero(np.diff(blk) != 0) + 1
    seg = np.split(np.arange(blk.size), edges)
    lens = np.array([step[s].sum() for s in seg])
    lens = lens[lens > 0]

    # corner angle between consecutive block directions
    dirs = []
    for s in seg:
        v = run.cnc[s[-1]] - run.cnc[s[0]]
        n = np.linalg.norm(v)
        if n > 1e-9:
            dirs.append(v / n)
    ang = np.array([])
    if len(dirs) > 1:
        dd = np.clip(np.einsum("ij,ij->i", np.array(dirs[:-1]), np.array(dirs[1:])), -1, 1)
        ang = np.degrees(np.arccos(dd))

    lo = np.nanpercentile(run.cnc, 1, axis=0)
    hi = np.nanpercentile(run.cnc, 99, axis=0)
    feeds = np.unique(np.round(run.feed[~g00]))
    all_feeds = np.unique(np.round(run.feed[np.isfinite(run.feed)]))
    return dict(
        run=name,
        n=len(run),
        # raw, classification-independent: this is what the rapid feed is
        # measured from (see detect_rapid_feed)
        feed_max=float(all_feeds.max()) if all_feeds.size else 0.0,
        feed_n_levels=int(all_feeds.size),
        duration_s=run.duration,
        dt_native_ms=run.dt_native * 1e3,
        resampled=int(run.resampled),
        n_invalid=int((~run.valid).sum()),
        n_blocks=int(np.unique(blk).size),
        g00_frac=float(g00.mean()),
        feed_cut_max=float(feeds.max()) if feeds.size else 0.0,
        feed_cut_med=float(np.median(run.feed[~g00])) if (~g00).any() else 0.0,
        speed_p95=float(np.percentile(speed, 95)),
        block_len_med_mm=float(np.median(lens)) if lens.size else 0.0,
        block_len_p10_mm=float(np.percentile(lens, 10)) if lens.size else 0.0,
        corner_ang_med_deg=float(np.median(ang)) if ang.size else 0.0,
        corner_ang_p90_deg=float(np.percentile(ang, 90)) if ang.size else 0.0,
        e_rms_x=float(np.sqrt((e[:, 0] ** 2).mean())) if e.size else 0.0,
        e_rms_y=float(np.sqrt((e[:, 1] ** 2).mean())) if e.size else 0.0,
        e_rms_z=float(np.sqrt((e[:, 2] ** 2).mean())) if e.size else 0.0,
        e_absmax=float(np.abs(e).max()) if e.size else 0.0,
        pos_x=float((lo[0] + hi[0]) / 2), pos_y=float((lo[1] + hi[1]) / 2),
        pos_z=float((lo[2] + hi[2]) / 2),
        span_mm=float(np.linalg.norm(hi - lo)),
    )


# --------------------------------------------------------------------------- #
# rapid-traverse feed: measured, not assumed
# --------------------------------------------------------------------------- #
MACHINE_FILE = "machine.json"
_machine_cache: Dict[str, object] = {}


def machine_path(cfg) -> Path:
    return Path(cfg.paths.meta_dir) / MACHINE_FILE


def detect_rapid_feed(per_run_max: Sequence[float], cfg,
                      feed_levels: Optional[Sequence[float]] = None):
    """Pick the rapid feed out of the per-run maximum feeds of a dataset.

    Returns ``(value_or_None, reason)``.

    Rapid traverse is a machine constant, so across many programs it shows up
    as *the most common* per-run maximum -- the mode, not the maximum, so that
    one odd program cannot move it. The candidate is rejected when it does not
    stand clearly above the next feed level in the dataset, because a guess
    that silently mislabels every G00 block is worse than being told.

    What this cannot see: a dataset with no rapid traverse at all. There the
    highest cutting feed looks exactly like a rapid. The G00 share printed by
    01_scan_runs is the check -- 0.00 or 0.90 means the answer is wrong.
    """
    vals = np.asarray([v for v in per_run_max if np.isfinite(v) and v > 0], float)
    if vals.size == 0:
        return None, "no usable feed values"

    levels, counts = np.unique(np.round(vals), return_counts=True)
    top = float(levels[int(np.argmax(counts))])
    share = float(counts.max()) / float(vals.size)
    n_above = int((vals > top + 0.5).sum())

    all_levels = np.unique(np.round(np.asarray(
        list(feed_levels) if feed_levels is not None else vals, float)))
    all_levels = all_levels[all_levels > 0]
    below = all_levels[all_levels < top]
    if below.size and top < cfg.data.rapid_feed_min_gap * float(below[-1]):
        return None, (f"{top:.0f} is only {top / float(below[-1]):.2f}x the next "
                      f"feed level ({below[-1]:.0f}); too close to call")

    reason = (f"mode of {vals.size} per-run maxima ({share:.0%} agree)"
              + (f"; {n_above} run(s) go higher and are ignored" if n_above else ""))
    return top, reason


def load_machine(cfg) -> dict:
    """Read work/meta/machine.json, or fall back to the configured constant."""
    key = str(machine_path(cfg))
    if key in _machine_cache:
        return _machine_cache[key]          # type: ignore[return-value]
    info = {"rapid_feed_mm_min": float(cfg.data.rapid_feed_mm_min),
            "source": "config"}
    p = machine_path(cfg)
    if cfg.data.rapid_feed_auto and p.is_file():
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            if saved.get("rapid_feed_mm_min"):
                info = {"rapid_feed_mm_min": float(saved["rapid_feed_mm_min"]),
                        "source": str(p), "n_runs": saved.get("n_runs"),
                        "feed_levels": saved.get("feed_levels")}
        except (ValueError, OSError):
            pass
    _machine_cache[key] = info
    return info


def save_machine(cfg, per_run_max: Sequence[float], feed_levels: Sequence[float]) -> dict:
    rapid, reason = detect_rapid_feed(per_run_max, cfg, feed_levels)
    info = {
        "rapid_feed_mm_min": rapid if rapid else float(cfg.data.rapid_feed_mm_min),
        "detected": rapid is not None,
        "reason": reason,
        "n_runs": len(list(per_run_max)),
        "feed_levels": sorted(float(f) for f in set(np.round(feed_levels))),
    }
    p = machine_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    _machine_cache.pop(str(p), None)
    return info


def rapid_feed(cfg) -> float:
    return float(load_machine(cfg)["rapid_feed_mm_min"])


def family_of(name: str) -> str:
    """``DOE_0042_RANDOMWALK3D_MIXED_F6000_S03`` -> ``RANDOMWALK3D_MIXED``."""
    import re
    s = re.sub(r"^DOE_\d+_?", "", name)
    s = re.sub(r"_F\d+.*$", "", s)
    return s or name


def feed_of(name: str) -> Optional[int]:
    """Programmed cutting feed taken from the folder name, if present."""
    import re
    m = re.search(r"_F(\d+)", name)
    return int(m.group(1)) if m else None
