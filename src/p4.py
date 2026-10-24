""" bayesian regression"""


import numpy as np
import matplotlib.pyplot as plt


from utils import (Report, OUTPUT_DIR, SEED, true_fun, dense_grid, save_fig, design_matrix, load_data)

DEGREE = 3
VAR = 0.01
BETA = 1/VAR
ALPHA = 0.05
LAM = ALPHA/BETA
Z95 = 1.96
N_GRID = 200
N_VALUES = (5, 10, 20, 50)



def posterior(Phi, t, alpha=ALPHA, beta=BETA):
    """Return (m_N, S_N). S_N^{-1} = alpha I + beta Phi^T Phi.
    m_N = beta S_N Phi^T t is obtained by solving S_N^{-1} m_N = beta Phi^T t;
    S_N by solving S_N^{-1} S_N = I"""
    M = Phi.shape[1]
    S_inv = alpha * np.eye(M) + beta * Phi.T @ Phi
    m_N = np.linalg.solve(S_inv, beta * Phi.T @ t)
    S_N = np.linalg.solve(S_inv, np.eye(M))
    return m_N, S_N


def predictive(Phi_star, m_N, S_N, beta=BETA):
    """m(x) = phi^T m_N ;  s^2(x) = 1/beta + phi^T S_N phi (row-wise)."""
    mean = Phi_star @ m_N
    var_param = np.einsum("ij,jk,ik->i", Phi_star, S_N, Phi_star)
    return mean, 1.0 / beta + var_param, var_param

def map_ridge(Phi, t, lam=LAM):
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ t)


def run(out_dir=OUTPUT_DIR / "p4"):
    rep = Report(out_dir, "P4: Bayesian Regression - MAP/Ridge Equivalence and Effect of N")
    rep.line(f"VAR = {VAR}, beta = {BETA:g}, alpha = {ALPHA}, lambda = alpha/beta = {LAM:g}, "
             f"degree = {DEGREE}, grid = {N_GRID} points on [0, 1]")
    xg = dense_grid(N_GRID)
    Pg = design_matrix(xg, DEGREE)

    x, t = load_data("p4_train_N10.csv")
    Phi = design_matrix(x, DEGREE)
    m_N, S_N = posterior(Phi, t)
    mean, var, _ = predictive(Pg, m_N, S_N)
    s = np.sqrt(var)
    w_map = map_ridge(Phi, t)
    y_map = Pg @ w_map

    rep.section("Deliverable 1: predictive distribution, p4_train_N10.csv")
    rep.array("m_N", m_N)
    rep.line("S_N =")
    for row in S_N:
        rep.line("   " + "  ".join(f"{v: .6e}" for v in row))
    rep.line(f"predictive std s(x): min = {s.min():.5f}, max = {s.max():.5f}; "
                 f"95% band half-width 1.96 s(x): min = {Z95 * s.min():.5f}, max = {Z95 * s.max():.5f}")
    
    rep.section("Deliverable 2: MAP / ridge equivalence")
    rep.array("w_MAP", w_map)
    rep.line(f"||w_MAP - m_N||_2 = {np.linalg.norm(w_map - m_N):.4e}")
    rep.line(f"max_x |w_MAP^T phi(x) - m(x)| over the grid = {np.max(np.abs(y_map - mean)):.4e}")
    rep.line("Both are the same vector analytically: beta (alpha I + beta Phi^T Phi)^{-1} Phi^T t "
                 "= (Phi^T Phi + (alpha/beta) I)^{-1} Phi^T t; the difference is floating-point round-off.")
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(xg, mean - Z95 * s, mean + Z95 * s, color="tab:red", alpha=0.2,
                        label=r"95% predictive interval $m(x) \pm 1.96\,s(x)$")
    ax.plot(xg, true_fun(xg), "g-", lw=1.5, label=r"$\sin(2\pi x)$")
    ax.plot(xg, mean, "r-", lw=2.2, label=r"predictive mean $m(x)$")
    ax.plot(xg, y_map, "k--", lw=1.4, label=r"MAP / ridge $w_{MAP}^\top\phi(x)$")
    ax.scatter(x, t, s=40, facecolors="none", edgecolors="b", label="training data (N = 10)", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(rf"P4.1-2  Bayesian cubic regression, $\alpha$ = {ALPHA}, $\beta$ = {BETA:g}  "
                     rf"($\|w_{{MAP}} - m_N\|_2$ = {np.linalg.norm(w_map - m_N):.1e})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    save_fig(fig, out_dir, "p4_1_2_predictive_and_map.png")


    rep.section("Deliverable 3: predictive interval width 2 * 1.96 * s(x) vs N")
    floor = 2 * Z95 / np.sqrt(BETA)
    rep.line(f"Irreducible-noise floor 2 * 1.96 * beta^(-1/2) = {floor:.5f}")
    rep.line(f"{'N':>4s} {'min width':>10s} {'mean width':>11s} {'max width':>10s} "
                 f"{'mean param var':>15s} {'mean width - floor':>19s}")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("viridis")
    summary = {}
    for k, N in enumerate(N_VALUES):
        xn, tn = load_data(f"p4_train_N{N}.csv")
        assert xn.size == N, f"p4_train_N{N}.csv has {xn.size} rows"
        mN, SN = posterior(design_matrix(xn, DEGREE), tn)
        _, v, vp = predictive(Pg, mN, SN)
        width = 2 * Z95 * np.sqrt(v)
        summary[N] = width
        rep.line(f"{N:4d} {width.min():10.5f} {width.mean():11.5f} {width.max():10.5f} "
                     f"{vp.mean():15.6e} {width.mean() - floor:19.5f}")
        ax.plot(xg, width, lw=2, color=cmap(k / (len(N_VALUES) - 1) * 0.9), label=f"N = {N}")
    ax.axhline(floor, color="k", ls="--", lw=1.2,
                   label=rf"noise floor $2\times1.96\,\beta^{{-1/2}}$ = {floor:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"interval width $2\times1.96\,s(x)$")
    ax.set_title("P4.3  95% predictive interval width vs. x for increasing N")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, out_dir, "p4_3_width_vs_N.png")
    
        # ---------------- Deliverable 4: discussion -------------------------
    rep.section("Deliverable 4: discussion")
    rep.line(
            "Each new observation adds beta phi(x_n) phi(x_n)^T to the posterior precision "
            "S_N^{-1} = alpha I + beta Phi^T Phi, so with more data the precision grows (roughly in "
            "proportion to N) and the posterior covariance S_N shrinks; the parameter-uncertainty term "
            "phi(x)^T S_N phi(x) therefore falls towards zero, which is why the band in p4_3_width_vs_N.png "
            f"narrows from a mean width of {summary[5].mean():.3f} at N = 5 to {summary[50].mean():.3f} at "
            "N = 50 (and is widest where data are sparse, near the ends of [0, 1]). The observation-noise "
            "term beta^{-1} does not depend on the data at all -- it is the variance of the noise on a new "
            f"target -- so the width can only approach the floor 2 x 1.96 x beta^(-1/2) = {floor:.3f}, never "
            "go below it, however large N becomes."
        )
    
    rep.save()
    return {"map_vs_mean": float(np.linalg.norm(w_map - m_N))}
    
    
if __name__ == "__main__":
    run()



