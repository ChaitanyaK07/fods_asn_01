import numpy as np
import matplotlib.pyplot as plt

from utils import (OUTPUT_DIR, load_data, Report, dense_grid, design_matrix, rmse, save_fig, true_fun)

D = 9
LAMS = np.logspace(-10, 2, 49)
LAM_GD = 1e-2
ETA = 0.04
TOL = 1e-6
MAX_ITERS = 5_000_000
NOISE_STD = np.sqrt(0.01)


def ridge_objective(phi, w, t, lam):
    r = phi@w - t
    return 0.5 * float(r@r) + 0.5*lam*float(w@w)

def ridge_closed_form(phi, t, lam):
    """to solve (Phi^T Phi + lam I) w = Phi^T t"""
    A = phi.T @ phi + lam * np.eye(phi.shape[1])
    return np.linalg.solve(A, phi.T @ t)


def ridge_gd(phi, t, lam, eta, w_ref, tol = TOL, max_iters=MAX_ITERS):
    """update_rule: w <- w - eta [Phi^T (Phi w - t) + lam w] from w = 0, stops when
    ||w - w_ref|| / ||w_ref|| < tol."""

    w = np.zeros(phi.shape[1])
    hist = [ridge_objective(phi, w, t, lam)]
    ref_norm = np.linalg.norm(w_ref)

    for k in range(1, max_iters+1):
        w = w - eta * (phi.T @ (phi@w - t) + lam*w)
        hist.append(ridge_objective(phi, w, t, lam))

        if np.linalg.norm(w-w_ref)/ref_norm < tol:
            return w, k, np.array(hist)
    raise RuntimeError(f"GD did not reach tol {tol} in {max_iters} iterations")


def run(out_dir = OUTPUT_DIR/"p2"):

    # setting_up

    rep = Report(out_dir, "P2: Ridge Regression as a MAP Estimate")

    x_train, t_train = load_data("poly_train.csv")
    x_test, t_test = load_data("poly_test.csv")

    phi = design_matrix(x_train, D)
    phi_test = design_matrix(x_test, D)

    rep.line(f"Train: poly_train.csv (N = {x_train.size}); Test: poly_test.csv (N = {x_test.size})")
    rep.line(f"d = {D}; lambda grid: np.logspace(-10, 2, 49)")
    rep.line("RMSE = sqrt(mean squared error) on the given set.")


    # deliverable 1 : closed form

    rep.section("Deliverable 1: closed-form ridge solution")
    W = np.array([ridge_closed_form(phi, t_train, lam) for lam in LAMS])

    train_rmse = np.array([rmse(phi, w, t_train) for w in W])
    test_rmse = np.array([rmse(phi_test, w, t_test) for w in W])

    rep.line(f"{'lambda':>12s} {'train RMSE':>12s} {'test RMSE':>12s} {'||w||_2':>14s}")
    for lam, w, a, b in zip(LAMS, W, train_rmse, test_rmse):
        rep.line(f"{lam:12.4e} {a:12.6f} {b:12.6f} {np.linalg.norm(w):14.6e}")


    #deliverable 2 : gd agreement at lambda = 10^-2

    rep.section(f"Deliverable 2: batch GD at lambda = {LAM_GD:g}, eta = {ETA}, w0 = 0")

    w_cf = ridge_closed_form(phi, t_train, LAM_GD)
    w_gd, n_iter, hist = ridge_gd(phi, t_train, LAM_GD, ETA, w_cf)
    rel = np.linalg.norm(w_gd - w_cf) / np.linalg.norm(w_cf)
    A0 = phi.T @ phi
    A = A0 + LAM_GD * np.eye(D + 1)
    c0, c1 = np.linalg.cond(A0), np.linalg.cond(A)
    eig = np.linalg.eigvalsh(A)


    rep.line(f"Iterations to reach ||w_GD - w_closed|| / ||w_closed|| < {TOL:g}: {n_iter}")
    rep.line(f"Final relative error = {rel:.4e}")
    rep.array("w_closed", w_cf)
    rep.array("w_GD    ", w_gd)
    rep.line(f"E~(w_closed) = {ridge_objective(phi, w_cf, t_train, LAM_GD):.10f}")
    rep.line(f"E~(w_GD)     = {hist[-1]:.10f}")
    rep.line(f"cond(Phi^T Phi)            = {c0:.4e}")
    rep.line(f"cond(Phi^T Phi + lambda I) = {c1:.4e}")
    rep.line(f"Eigenvalues of Phi^T Phi + lambda I: min = {eig[0]:.4e}, max = {eig[-1]:.4e} "
                 f"(stability requires eta < 2/max = {2 / eig[-1]:.4f}; eta = {ETA})")
    rep.line(f"The penalty lifts every eigenvalue of Phi^T Phi by lambda = {LAM_GD:g}, which raises the "
                 f"near-zero smallest eigenvalue to about lambda and cuts the condition number from "
                 f"{c0:.1e} to {c1:.1e} (about {c0 / c1:.0e} times smaller), making plain GD practical.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(np.arange(hist.size), hist, lw=1.8, label=r"$\tilde E(w_k)$ (GD)")
    ax.axhline(ridge_objective(phi, w_cf, t_train, LAM_GD), color="r", ls="--", lw=1,
                   label=r"$\tilde E(w_{closed})$")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, hist.size - 1)
    ax.set_xlabel("iteration (symlog scale)")
    ax.set_ylabel(r"$\tilde E(w)$ (log scale)")
    ax.set_title(rf"P2.2  Ridge GD, $\lambda$ = {LAM_GD:g}, $\eta$ = {ETA}: "
                     f"{n_iter} iterations to rel. error < {TOL:g}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, out_dir, "p2_2_ridge_gd_objective.png")



    # deliverable 3 : 

    rep.section("Deliverable 3: RMSE sweep and lambda*")
    train_rmse = []
    test_rmse = []
    for lam in LAMS:
        w = ridge_closed_form(phi, t_train, lam)
        train_rmse.append(rmse(phi, w , t_train))
        test_rmse.append(rmse(phi_test, w, t_test))
    train_rmse = np.array(train_rmse)
    test_rmse = np.array(test_rmse)

    best = np.argmin(test_rmse)
    lam_star = LAMS[best]

    min_rmse = test_rmse[best]
    rep.line(f"lambda* = {lam_star:.4e}")
    rep.line(f"test RMSE at lambda* = {min_rmse:.6f}")

    limit = 1.05 * min_rmse
    rep.line(f"1.05 x min test RMSE = {limit:.6f}")

    left = best

    while left > 0 and test_rmse[left-1] <= limit:
        left -=1

    right = best

    while right < len(LAMS) - 1 and test_rmse[right + 1] <= limit:
        right +=1

    rep.line(f"interval: [{LAMS[left]:.4e}, {LAMS[right]:.4e}]")

    
    for i in range(len(LAMS)):
        if test_rmse[i] <= limit and (i < left or i > right):
            rep.line(f"  note: lambda = {LAMS[i]:.4e} also passes (test RMSE {test_rmse[i]:.6f}), "
                 f"but is separated from the interval")


    rep.line(
    "The minimum is flat: test RMSE stays within 5% of its minimum for almost two decades "
    "of lambda, so the exact choice inside this interval hardly matters. "
    "To the left the penalty is too weak and the degree-9 polynomial overfits (train RMSE "
    "near 0, test RMSE rising), while to the right the penalty shrinks the weights too much "
    "and the fit underfits (train and test RMSE both rise together)."
    )


    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(LAMS, train_rmse, "o-", label="train RMSE")
    ax.plot(LAMS, test_rmse, "s-", label="test RMSE")
    ax.axhline(0.1, color="k", linestyle="--", label="noise level sqrt(0.01) = 0.1")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("RMSE")
    ax.set_title("P2.3 Ridge: train and test RMSE vs lambda")   
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir, "p2_3_rmse_sweep.png")

# ======================= Deliverable 4 =======================
    rep.section("Deliverable 4: fitted curves")

    x_plot = dense_grid(400)
    Phi_plot = design_matrix(x_plot, D)

    cases = [
        (1e-10, "overfit"),
        (lam_star, "lambda*"),
        (1e-2, "underfit"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (lam, name) in zip(axes, cases):
        w = ridge_closed_form(phi, t_train, lam)
        w_norm = np.linalg.norm(w)
        rep.line(f"{name}: lambda = {lam:.4e}, ||w||_2 = {w_norm:.6e}")

        ax.plot(x_plot, true_fun(x_plot), "g--", label="sin(2 pi x)")   # plotting only
        ax.plot(x_plot, Phi_plot @ w, "r-", label="ridge fit")
        ax.scatter(x_train, t_train, facecolors="none", edgecolors="b", label="training data")
        ax.set_title(f"{name}: lambda = {lam:.2e}, ||w|| = {w_norm:.3g}")
        ax.set_xlabel("x")
        ax.set_ylim(-1.6, 1.6)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("y")
    axes[0].legend()
    save_fig(fig, out_dir, "p2_4_fitted_curves.png")


    rep.save()
    return {"lam_star": lam_star}

if __name__ == "__main__":
    run()

    

    




    







    











