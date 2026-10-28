"""Table 1: checks on the transformation and window functions.

Uses the same functions the experiments call (q2.data).
Run: python -m q2.checks
"""
import json
from pathlib import Path

import torch

from q2.data import make_windows, mirror, rotate

OUT = Path("results/tables")


def bright_pixel(video):
    frame, _, row, col = (video == 1).nonzero()[0].tolist()
    return frame, row, col


def window_check(k):
    """Video whose frame f (1-based) is filled with the value f; window at t=1."""
    video = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64)
    x, y, mask = make_windows(video[None], torch.tensor([1]), torch.tensor([k]))
    inputs = [int(v) for v in x[0, :, 0, 0, 0]]
    targets = [int(v) for v, m in zip(y[0, :, 0, 0, 0], mask[0]) if m]
    return inputs, targets


def main():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1

    rows = []
    for name, lhs, rhs in [
        ("(a) max|M(M(x)) - x|", mirror(mirror(x)), x),
        ("(b) max|R^4(x) - x|", rotate(rotate(rotate(rotate(x)))), x),
        ("(c) max|MRM(x) - R^-1(x)|", mirror(rotate(mirror(x))), rotate(x, -1)),
    ]:
        diff = (lhs - rhs).abs().max().item()
        rows.append({"check": name, "expected": "0",
                     "yours": f"{diff:g} (torch.equal: {torch.equal(lhs, rhs)})",
                     "pass": diff == 0 and torch.equal(lhs, rhs)})

    for name, video, expected in [
        ("(d) bright pixel of M(p)", mirror(p), (7, 10, 43)),
        ("(e) bright pixel of R(p)", rotate(p), (7, 43, 10)),
    ]:
        got = bright_pixel(video)
        rows.append({"check": name + " (frame, row, column)", "expected": str(expected),
                     "yours": str(got), "pass": got == expected})

    for name, k, expected in [
        ("(f) 1x input / target frames", 1, (list(range(1, 6)), list(range(6, 11)))),
        ("(g) 2x input / target frames", 2, ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19])),
        ("(h) 3x input / target frames", 3, ([1, 4, 7, 10, 13], [16, 19])),
    ]:
        got = window_check(k)
        rows.append({"check": name, "expected": f"{expected[0]} / {expected[1]}",
                     "yours": f"{got[0]} / {got[1]}", "pass": got == expected})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table1_checks.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<40} expected {r['expected']:<45} yours {r['yours']}")
    if not all(r["pass"] for r in rows):
        raise SystemExit("Table 1 has failing rows: fix q2/data.py before continuing.")


if __name__ == "__main__":
    main()
