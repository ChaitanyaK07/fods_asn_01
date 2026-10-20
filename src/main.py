import sys
import time

import p1
import p2
import p3
import p4
from utils import OUTPUT_DIR

PROBLEMS = {"p1": p1.run, "p2": p2.run, "p3": p3.run, "p4": p4.run}


def main(selected):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in selected:
        start = time.perf_counter()
        print(f"\n{'#' * 70}\n# Running {name.upper()}\n{'#' * 70}")
        PROBLEMS[name](OUTPUT_DIR / name)
        print(f"\n[{name.upper()} done in {time.perf_counter() - start:.1f}s -> {OUTPUT_DIR / name}]")

    # one combined text file with all results
    parts = []
    for name in PROBLEMS:
        f = OUTPUT_DIR / name / "results.txt"
        if f.exists():
            parts.append(f.read_text(encoding="utf-8"))
    (OUTPUT_DIR / "all_results.txt").write_text("\n\n".join(parts), encoding="utf-8")
    print(f"\nCombined results written to {OUTPUT_DIR / 'all_results.txt'}")


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]] or list(PROBLEMS)
    unknown = [a for a in args if a not in PROBLEMS]
    if unknown:
        sys.exit(f"Unknown problem(s): {unknown}. Choose from {list(PROBLEMS)}.")
    main(args)