# CS F320 – Foundations of Data Science, Assignment 1

Polynomial regression, ridge, lasso and Bayesian linear regression, implementation

| Problem | Topic | File |
|---|---|---|
| P1 | Polynomial regression: closed form vs. Batch / SGD / Mini-batch GD (QR-reparametrised, d = 9), learning-rate study | `p1.py` |
| P2 | Ridge regression as a MAP estimate: closed form, plain GD, RMSE sweep, fitted curves | `p2.py` |
| P3 | Lasso via subgradient descent (Legendre basis), optimality check, Lasso vs. Ridge | `p3.py` |
| P4 | Bayesian regression: posterior/predictive, MAP–ridge equivalence, effect of N | `p4.py` |

## Folder structure
```
project/
├── data/
│   ├── poly_train.csv
│   ├── poly_test.csv
│   ├── p4_train_N5.csv
│   ├── p4_train_N10.csv
│   ├── p4_train_N20.csv
│   └── p4_train_N50.csv
└── src/
    ├── main.py      # runs one or more problems and combines results
    ├── utils.py     # data loading, design matrices, metrics, Report, plotting helpers
    ├── p1.py
    ├── p2.py
    ├── p3.py
    ├── p4.py
    └── output/      # created automatically on first run
```

All CSVs have columns `x,y` and are loaded as provided (no data is generated or modified). `sin(2πx)` is used for plotting only.

> `utils.py` resolves paths relative to its own location. For the layout above, `DATA_DIR` must be
> `ROOT_DIR.parent / "data"`. If you keep `data/` inside `src/` instead, no change is needed.

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`

```powershell
pip install numpy matplotlib
```

## Running

All files must be in the same folder (`src/`). Run from `src/`:

```powershell
cd src

# run everything (P1–P4)
python main.py

# run selected problems
python p1.py
python p2.py
python p3.py
python p4.py
```

Paths are resolved from the file location, so the working directory does not matter.

## Output

Written to `src/output/`:

```
output/
├── p1/  results.txt + p1_1_closed_form_fits.png, p1_3_loss_curves.png, p1_4_learning_rate_study.png
├── p2/  results.txt + p2_2_ridge_gd_objective.png, p2_3_rmse_sweep.png, p2_4_fitted_curves.png
├── p3/  results.txt + p3_1_convergence.png, p3_2_rmse_sweep.png, p3_3_coefficient_path.png, p3_6_fitted_curves.png
├── p4/  results.txt + p4_1_2_predictive_and_map.png, p4_3_width_vs_N.png
└── all_results.txt      # concatenation of every results.txt
```

Each `results.txt` is organised by deliverable, mirroring the assignment. The same text is printed to the console.

Running a file standalone (e.g. `python p1.py`) writes to `output/P1` / `output/P3` (uppercase) for P1 and P3 and `output/p2` / `output/p4` for P2 and P4. `main.py` always uses lowercase folder names.

## Key settings

| Item | Value |
|---|---|
| Random seed (SGD / mini-batch shuffling) | `SEED = 42` (`utils.py`) |
| P1 | d = 9, η = 0.1, 2000 epochs, mini-batch size 3; LR study: η ∈ {0.01, 0.5, 2.1} plus η = 2 boundary, 100 epochs |
| P2 | d = 9 monomial basis, `λ ∈ np.logspace(-10, 2, 49)`, GD at λ = 1e-2 with η = 0.04, stop at relative error < 1e-6 |
| P3 | Shifted Legendre basis (d = 9), η = 0.05, 20,000 iterations, `λ ∈ np.logspace(-4, 0, 21)`, "≈ 0" means `|w_j| < 2ηλ` |
| P4 | Degree 3, α = 0.05, β = 100 (σ² = 0.01), N ∈ {5, 10, 20, 50} |

## Implementation notes

- **P1:** the closed form uses `np.linalg.lstsq`. GD runs on the thin-QR problem `E(α) = ½‖Qα − t‖²` with `α = Rw`, and weights are recovered via `w = R⁻¹α`.
- **P2–P4:** all linear systems are solved with `np.linalg.solve`; no explicit inverses are used for the ridge/MAP solutions.
- **P3:** subgradient descent is not monotone, so the best-so-far objective is tracked for the convergence plot only. All other quantities use the final iterate.
- **P3 reference check:** results at λ = 0.1 are compared against the supplied `E* = 0.2625` and `w*`.
