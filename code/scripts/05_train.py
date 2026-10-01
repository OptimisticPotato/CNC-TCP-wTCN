"""Main training run. SPEC section 7 step 3.

Programs are split program-wise (never window-wise), resample augmentation is
on, the loss is the three-term one, and the first receptive field of every crop
is excluded from it.

    python scripts/05_train.py --selection r1
    python scripts/05_train.py --selection r1 --set train.name=tcn_rf250 \
        --set model.dilations=1,2,4,8,16,32,64,128
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from config import add_config_args, as_dict, configure


def read_selection(cfg, tag: str):
    path = cfg.paths.meta_dir / f"selection_{tag}.txt"
    if not path.is_file():
        raise SystemExit(f"no such selection: {path}\nrun scripts/02_select_doe.py first")
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", default="r1")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--resume-split", default=None,
                    help="reuse the split.json of an earlier run")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    from tcn_cnc2tcp import dataset as ds
    from tcn_cnc2tcp.plots import training_curve
    from tcn_cnc2tcp.trainer import train_model

    names = args.runs or read_selection(cfg, args.selection)
    out = cfg.paths.runs_out_dir / cfg.train.name
    out.mkdir(parents=True, exist_ok=True)

    if args.resume_split:
        split = ds.load_split(args.resume_split)
    else:
        split = ds.split_programs(cfg, names)
    ds.save_split(out / "split.json", split)

    print(cfg.describe())
    print(f"\nprograms: {len(names)}  ->  train {len(split['train'])} / "
          f"val {len(split['val'])} / test {len(split['test'])}")
    print("val :", ", ".join(split["val"]))
    print("test:", ", ".join(split["test"]))

    print("\ncaching / loading")
    train_programs = ds.load_programs(cfg, split["train"], verbose=False)
    val_programs = ds.load_programs(cfg, split["val"], verbose=False)
    n_samples = sum(len(p) for p in train_programs)
    print(f"  train {n_samples:,} samples "
          f"({n_samples * cfg.data.dt / 60:.1f} min of machining)")

    (out / "config.json").write_text(json.dumps(as_dict(cfg), indent=2), encoding="utf-8")

    log_path = out / "train.log"
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n=== {cfg.train.name} ===")
    res = train_model(cfg, train_programs, val_programs, out, log=log,
                      tag=cfg.train.name)
    log_file.close()

    training_curve(out / "training_curve.png", res["history"])
    print(f"\nbest checkpoint : {res['best_path']}")
    print(f"best score      : {res['best']:.4f} um")
    print(f"next            : python scripts/06_fir_baseline.py --run {cfg.train.name}")
    print("                  then 07_evaluate.py, then 08_compare_fir_tcn.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
