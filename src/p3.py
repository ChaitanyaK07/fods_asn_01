"""L1-regularised regression via subgradient descent: prob-3"""


import numpy as np
import matplotlib.pyplot as plt

from utils import (true_fun, OUTPUT_DIR, SEED, rmse, Report, dense_grid, load_data, legendre_design, save_fig)

D = 9
ETA = 0.05
N_ITERS = 20_000
LAMS = np.logspace(-4, 0, 21)
lam_check = 0.1
E_STAR = 0.2625
W_STAR = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0])
thresh = 0.01
NOISE_STD = np.sqrt(0.01)



def lasso_obj(phi, w, t, lam):
    r = phi@w - t
    return 0.5 * float(r@r) + lam*float(np.sum(np.abs(w)))

def lasso_subgrad(phi, t, lam,eta=ETA, track=False, n_iters = N_ITERS):
    w = np.zeros(phi.shape[1])

    if track:
        E_hist = np.zeros(n_iters+1)
        E_hist[0] = lasso_obj(phi, w, t, lam)
    for k in range(1,n_iters+1):
        g = phi.T @ (phi@w - t) + lam*np.sign(w)
        w = w - eta*g
        if track:
            E_hist[k] = lasso_obj(phi, w, t, lam)
    if track:
        return w, E_hist, np.minimum.accumulate(E_hist)
    return w


def near_zero_mask(w, lam, eta=ETA):
    '''is zero if |w| < 2(eta)(lam)'''
    return np.abs(w) < 2*eta*lam

def ridge_closed_form(phi, t, lam):
    A = phi.T @ phi + lam*(np.eye(phi.shape[1]))
    return np.linalg.solve(A, phi.T@t)


def run(output_dir = OUTPUT_DIR/"P3"):
    rep = Report(output_dir, "P3: L1-Regularised Regression via Subgradient Descent")
    x, t = load_data("poly_train.csv")
    x_test, t_test = load_data("poly_test.csv")

    phi = legendre_design(x, D)
    phi_test = legendre_design(x_test, D)

    rep.line(f"Train: poly_train.csv (N = {x.size}); Test: poly_test.csv (N = {x_test.size})")
    rep.line(f"  max |Phi - np.polynomial.legendre.legvander(2x-1, 9)| = "
                 f"{np.max(np.abs(phi - np.polynomial.legendre.legvander(2 * x - 1, D))):.2e}")
    rep.line(f"  cond(Phi^T Phi) = {np.linalg.cond(phi.T @ phi):.4e}")
    rep.line(f"eta = {ETA}, w0 = 0, {N_ITERS} iterations per lambda")

    # convergence and verification at lambda = 0.1 - deliverable 1

    rep.section(f"Deliverable 1: convergence and verification at lambda = {lam_check}")

    w, E_hist, E_best = lasso_subgrad(phi, t, lam_check, track=True)

    E_final = E_hist[-1]
    rep.line(f"(a) E(final iterate) = {E_final:.6f}")
    rep.array("final w", w, fmt="{: .5f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    it = np.arange(N_ITERS + 1)
    ax.semilogy(it, E_hist, lw=1.0, alpha=0.8, label="E(w_k)")
    ax.semilogy(it, E_best, "--", lw=1.6, color="k", label="best-so-far E")
    ax.axhline(E_STAR, color="r", ls=":", lw=1.2, label=rf"reference $E^\star$ = {E_STAR}")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, N_ITERS)
    ax.set_xlabel("iteration (symlog scale)")
    ax.set_ylabel("E(w) (log scale)")
    ax.set_title(rf"P3.1a  Lasso subgradient descent, $\lambda$ = {lam_check}, $\eta$ = {ETA}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, output_dir, "p3_1_convergence.png")

    # deliverable 1.b - optimality conditions

    r = t - phi@w
    corr = phi.T@r

    z = near_zero_mask(w, lam_check)
    nz = ~z
    kkt_nz = np.max(np.abs(corr[nz] - lam_check * np.sign(w[nz]))) if nz.any() else float("nan")
    kkt_z = np.max(np.abs(corr[z]) / lam_check) if z.any() else float("nan")

    rep.line(f"(b) ~0 threshold 2*eta*lambda = {2 * ETA * lam_check:.4f};  "
                 f"~0 indices: {np.where(z)[0].tolist()};  non-zero indices: {np.where(nz)[0].tolist()}")
    rep.array("    Phi_j^T r", corr, fmt="{: .5f}")
    rep.line(f"    max |Phi_j^T r - lambda sign(w_j)| over non-zero coefficients = {kkt_nz:.4e}")
    rep.line(f"    max |Phi_j^T r| / lambda over ~0 coefficients               = {kkt_z:.4f}  "
                 f"({'<= 1: condition satisfied' if kkt_z <= 1 else '> 1: condition violated'})")

    rep.line(f"(c) reference E* = {E_STAR};  E(final) - E* = {E_final - E_STAR:.4e}")
    rep.array("    w*", W_STAR, fmt="{: .3f}")
    rep.array("    w (rounded)", np.round(w, 3), fmt="{: .3f}")
    rep.line(f"    ||w - w*||_2 = {np.linalg.norm(w - W_STAR):.4e}  "
                 f"(chatter amplitude ~ eta*lambda = {ETA * lam_check:.3f}; w* itself is rounded to 3 dp)")


    # sweep over grid

    W = np.array([lasso_subgrad(phi, t, lam) for lam in LAMS])
    train = np.array([rmse(phi,w, t) for w in W])
    test = np.array([rmse(phi_test, w, t_test) for w in W])
    nzero = np.array([int(near_zero_mask(w, lam).sum()) for w, lam in zip(W, LAMS)])

    rep.section("Deliverable 2: train / test RMSE vs lambda (Lasso)")
    i_star = int(np.argmin(test))
    lam_star = LAMS[i_star]
    rep.line(f"Minimum test RMSE {test[i_star]:.6f} at lambda = {lam_star:.4e}; noise reference 0.1.")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogx(LAMS, train, "o-", ms=4, label="train RMSE")
    ax.semilogx(LAMS, test, "s-", ms=4, label="test RMSE")
    ax.axhline(NOISE_STD, color="k", ls="--", lw=1, label=r"noise level $\sqrt{0.01}$ = 0.1")
    ax.axvline(lam_star, color="r", ls=":", lw=1.2, label=rf"$\lambda^\star$ = {lam_star:.2e}")
    ax.set_xlabel(r"$\lambda$ (log scale)")
    ax.set_ylabel("RMSE")
    ax.set_title("P3.2  Lasso (Legendre basis, d = 9): train / test RMSE vs. $\\lambda$")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, output_dir, "p3_2_rmse_sweep.png")

    #deliverable 3

    rep.section("Deliverable 3: coefficient path and number of ~0 coefficients")
    rep.line(f"{'lambda':>11s} {'train':>9s} {'test':>9s} {'#~0':>4s}  ~0 indices          "
                 + " ".join(f"{'w' + str(j):>8s}" for j in range(D + 1)))
    for lam, w_, a, b, k in zip(LAMS, W, train, test, nzero):
            idx = str(np.where(near_zero_mask(w_, lam))[0].tolist())
            rep.line(f"{lam:11.4e} {a:9.5f} {b:9.5f} {k:4d}  {idx:20s}"
                     + " ".join(f"{v:8.4f}" for v in w_))

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for j in range(D + 1):
            ax.semilogx(LAMS, W[:, j], "o-", ms=3, color=cmap(j), label=rf"$w_{{{j}}}$")
    ax.fill_between(LAMS, -2 * ETA * LAMS, 2 * ETA * LAMS, color="gray", alpha=0.3,
                        label=r"'$\approx 0$' band $|w_j| < 2\eta\lambda$")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(lam_star, color="r", ls=":", lw=1.2, label=r"$\lambda^\star$")
    ax.set_xlabel(r"$\lambda$ (log scale)")
    ax.set_ylabel(r"$w_j$")
    ax.set_title("P3.3  Lasso coefficient path (Legendre basis)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=2, fontsize=8, loc="best")
    save_fig(fig, output_dir, "p3_3_coefficient_path.png")


    # deliverable - 4 : lambda *

    rep.section("Deliverable 4: selection of lambda* (Lasso)")
    w_l = W[i_star]
    z_star = np.where(near_zero_mask(w_l, lam_star))[0]
    rep.line(f"lambda* = {lam_star:.4e}  (grid index {i_star})")
    rep.line(f"test RMSE at lambda* = {test[i_star]:.6f};  train RMSE = {train[i_star]:.6f}")

    rep.line(f"~0 threshold at lambda*: 2*eta*lambda* = {2 * ETA * lam_star:.4e}")
    rep.line(f"number of ~0 coefficients at lambda* = {z_star.size};  indices = {z_star.tolist()}")
    rep.array("w(lambda*)", w_l, fmt="{: .5f}")

    # deliverable 5: comparing lasso and ridge

    rep.section("Deliverable 5: Lasso vs Ridge")

    W_r = np.array([ridge_closed_form(phi, t, lam) for lam in LAMS])
    test_r = np.array([rmse(phi_test, w, t_test) for w in W_r])
    train_r = np.array([rmse(phi, w, t) for w in W_r])

    j_r = int(np.argmin(test_r))
    lam_r = LAMS[j_r]
    w_r = W_r[j_r]

    rep.line("Ridge sweep (closed form, linear solve):")
    rep.line(f"{'lambda':>11s} {'train':>9s} {'test':>9s} {'||w||_2':>9s}")
    for lam, a, b, w_ in zip(LAMS, train_r, test_r, W_r):
        rep.line(f"{lam:11.4e} {a:9.5f} {b:9.5f} {np.linalg.norm(w_):9.4f}")
    rep.line(f"Ridge best lambda = {lam_r:.4e} (grid index {j_r})")
    rep.array("w_ridge(best)", w_r, fmt="{: .5f}")
    rep.line("")
    rep.line(f"{'method':8s} {'best lambda':>12s} {'test RMSE':>10s} {'||w||_2':>9s} "
                 f"{'#|w_j|<0.01':>12s}  indices")
    table = {}
    for name, lam, w_, e in (("Lasso", lam_star, w_l, test[i_star]), ("Ridge", lam_r, w_r, test_r[j_r])):
        small = np.where(np.abs(w_) < thresh)[0]
        table[name] = (lam, e, np.linalg.norm(w_), small.size)
        rep.line(f"{name:8s} {lam:12.4e} {e:10.6f} {np.linalg.norm(w_):9.4f} {small.size:12d}  "
                     f"{small.tolist()}")

    #deliverable-6 : fitted curves

    xg = dense_grid(400)
    Pg = legendre_design(xg, D)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xg, true_fun(xg), "g--", lw=1.2, label=r"$\sin(2\pi x)$")
    ax.plot(xg, Pg @ w_l, "r-", lw=2, label=rf"Lasso, $\lambda^\star$ = {lam_star:.2e}")
    ax.plot(xg, Pg @ w_r, "b-.", lw=2, label=rf"Ridge, best $\lambda$ = {lam_r:.2e}")
    ax.scatter(x, t, s=40, facecolors="none", edgecolors="k", label="training data", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("P3.6  Lasso vs. Ridge fits (Legendre basis, d = 9)")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, output_dir, "p3_6_fitted_curves.png")

    #deliverable-7 

    rep.section("Deliverable 7: discussion")
    lasso_small = table["Lasso"][3]
    ridge_small = table["Ridge"][3]
    i_big = LAMS.size - 1
    rep.line(
            "The coefficient path (p3_3_coefficient_path.png) shows the L1 penalty driving coefficients "
            f"into the '~0' band one after another -- {nzero[0]} ~0 coefficient(s) at lambda = 1e-4, "
            f"{z_star.size} at lambda*, and {nzero[i_big]} at lambda = 1 -- whereas at their best lambda "
            f"Ridge has {ridge_small} coefficient(s) with |w_j| < 0.01 against {lasso_small} for Lasso "
            "(Deliverable 5 table). This follows from the optimality conditions: the L1 subgradient at "
            "w_j = 0 is the whole interval [-lambda, lambda], so w_j = 0 is optimal whenever "
            "|Phi_j^T r| <= lambda, i.e. once a feature's correlation with the residual falls below lambda "
            "the coefficient is held at exactly zero (the SSE contours touch the l1 diamond at a corner on "
            "an axis); the L2 penalty's gradient lambda w_j vanishes at w_j = 0, so it only rescales "
            "each coefficient towards zero along the circular ball and never sets it exactly to zero. "
            f"As lambda grows past lambda* = {lam_star:.2e}, more coefficients are removed and the rest "
            f"are shrunk, so train RMSE rises monotonically (from {train[i_star]:.3f} to {train[i_big]:.3f} at "
            f"lambda = 1) and test RMSE also rises (from {test[i_star]:.3f} to {test[i_big]:.3f}): the model "
            "moves from a good fit into underfitting, with both errors climbing well above the 0.1 noise level."
        )



    rep.save()
    return {"lam_star": lam_star, "lam_ridge": lam_r}

if __name__ == "__main__":
    run()




    




















    







