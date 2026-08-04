"""Run isolated FP8 launch-order trace trials sequentially."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--log-dir", type=Path, default=Path("scratchpad/order_trial_logs"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    orders = ("sfa-first", "weight-first")
    for trial in range(1, args.trials + 1):
        trial_orders = orders if trial % 2 else tuple(reversed(orders))
        for order in trial_orders:
            command = [
                sys.executable,
                "fp8_trace.py",
                "--up-proj-order",
                order,
                "--profile-mode",
                "fwd",
                "--warmup",
                str(args.warmup),
                "--run-index",
                str(trial),
            ]
            log_path = args.log_dir / f"{order.replace('-', '_')}_run{trial}.log"
            print(f"[{trial}/{args.trials}] {order}", flush=True)
            with log_path.open("w") as log:
                subprocess.run(
                    command,
                    cwd=Path(__file__).parent,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )


if __name__ == "__main__":
    main()
