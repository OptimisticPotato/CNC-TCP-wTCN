"""Sanity check: python, torch/CUDA, paths, and what the config resolves to.

    python scripts/00_check_env.py                # environment + paths
    python scripts/00_check_env.py --selftest     # ... and exercise the model

``--selftest`` runs the whole torch path on synthetic data: build the model,
verify the two structural guarantees (zero input -> zero output, no future
leak), forward, three-term loss, backward, optimiser step, and a chunked
whole-sequence prediction. Run it once on a new machine before 04/05.
"""
from __future__ import annotations

import argparse
import platform
import sys

import _bootstrap  # noqa: F401
from config import add_config_args, configure


def selftest(cfg) -> int:
    import numpy as np
    import torch

    from tcn_cnc2tcp import dataset as ds, features
    from tcn_cnc2tcp.losses import TCNLoss
    from tcn_cnc2tcp.model import build_model, dilations_for_ms
    from tcn_cnc2tcp.trainer import make_grad_scaler, pick_device

    device = pick_device()
    print(f"\n--- selftest on {device} -----------------------------------------")
    model = build_model(cfg).to(device)
    print(f"parameters        : {model.num_parameters():,}")
    print(f"receptive field   : {model.receptive_field} samples "
          f"({model.receptive_field * cfg.data.dt * 1e3:.1f} ms)")

    zero = model.check_zero_input(device=device)
    leak = model.check_causality(device=device)
    homo = model.check_homogeneity(device=device)
    print(f"zero in -> zero out: {zero:.3e}   (must be 0)")
    print(f"future leak        : {leak:.3e}   (must be 0)")
    print(f"scale non-linearity: {homo:.3e}   (0 = f(2u)=2f(u) exactly, so "
          f"amplitude saturation is NOT representable; non-zero with tanh)")
    if zero != 0.0 or leak != 0.0:
        print("FAILED: the model breaks a structural guarantee")
        return 1

    # a synthetic program: a step move, then a reversal
    n = max(4096, 3 * cfg.receptive_field)
    t = np.arange(n) * cfg.data.dt
    cnc = np.zeros((n, 3))
    cnc[:, 0] = 10.0 * np.sin(2 * np.pi * 3.0 * t)
    cnc[:, 1] = np.clip((t - t[n // 3]) * 50.0, 0, 5.0)
    g00 = np.zeros(n, bool)
    g00[: n // 8] = True
    u = features.build_inputs(cnc, g00, cfg.increment_scale)
    ident = features.check_identity(u, cnc, cfg.increment_scale)
    print(f"channel identity   : {ident:.3e}   (float32 rounding, ~1e-7)")

    e = np.zeros((n, 3), np.float32)
    e[:, 0] = np.float32(3.0 * np.gradient(np.gradient(cnc[:, 0])))
    prog = ds.Program(name="synthetic", cnc=cnc, e=e, u=u, g00=g00,
                      valid=np.ones(n, bool), block=np.zeros(n, np.int32),
                      dt=cfg.data.dt)

    loader = ds.make_loader(cfg, [prog], augment=True, shuffle=False)
    batch = next(iter(loader))
    ub, eb, wb = (x.to(device) for x in batch)
    print(f"batch shapes       : u {tuple(ub.shape)}  e {tuple(eb.shape)}  "
          f"w {tuple(wb.shape)}  (scored samples {int((wb > 0).sum())})")

    crit = TCNLoss(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    use_amp = bool(cfg.train.amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=use_amp):
            pred = model(ub)
        loss, parts = crit(pred.float(), eb, wb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        print(f"step {step}             : " +
              "  ".join(f"{k}={v:.4f}" for k, v in parts.items()))
    if not torch.isfinite(loss):
        print("FAILED: loss is not finite")
        return 1

    full = ds.predict_program(model, cfg, prog, device, chunk=1024)
    print(f"chunked prediction : {full.shape}, finite={bool(np.isfinite(full).all())}")

    d = dilations_for_ms(200.0, cfg.data.dt, cfg.model.kernel_size)
    print(f"dilations for 200ms: {d} -> "
          f"{1 + (cfg.model.kernel_size - 1) * sum(d)} samples")
    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="also exercise model / loss / dataset on synthetic data")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = configure(args)

    print("python          :", sys.version.split()[0], platform.platform())
    try:
        import numpy, pandas, scipy, matplotlib  # noqa: F401
        print("numpy/scipy     :", numpy.__version__, "/", scipy.__version__)
        print("pandas/mpl      :", pandas.__version__, "/", matplotlib.__version__)
    except ImportError as exc:
        print("MISSING package :", exc)
        return 1

    have_torch = True
    try:
        import torch
    except ImportError:
        have_torch = False
        print("torch           : MISSING -- run setup_venv.ps1 "
              "(everything below still applies)")

    if have_torch:
        print("torch           :", torch.__version__)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"cuda            : {torch.version.cuda}  {p.name}  "
                  f"{p.total_memory / 2**30:.1f} GiB")
        else:
            print("cuda            : NOT available -- training would run on the CPU")

    print()
    print(cfg.describe())

    from tcn_cnc2tcp.runs_io import list_runs, read_exclude_file
    try:
        names = list_runs(cfg)
    except Exception as exc:                       # noqa: BLE001
        print("\nrun discovery FAILED:", exc)
        return 1
    ex = read_exclude_file(cfg.paths.exclude_file)
    print(f"\nrun folders     : {len(names)} usable, {len(ex)} excluded "
          f"({cfg.paths.exclude_file.name})")
    if names:
        print("first / last    :", names[0], "...", names[-1])

    if not have_torch:
        print("\ntorch is missing, so the model path cannot be checked. "
              "Install it with setup_venv.ps1, then re-run with --selftest.")
        return 1
    if args.selftest:
        return selftest(cfg)
    print("\n(run with --selftest to exercise the model/loss/dataset path)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
