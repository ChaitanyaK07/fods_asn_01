import numpy as np
import matplotlib.pyplot as plt

from utils import (SEED, OUTPUT_DIR, dense_grid, design_matrix, Report, load_data, save_fig, sse, true_fun)

D_MAX = 9
epochs = 2000
batch_size = 3
eta_main = 0.1

eta_study = {"too_small" : 0.01, "well_chosen" : 0.5, "too_large" : 2.1}
epoch_study = 100
eta_boundary = 2.0


def closed_form(phi, t):
    w, *_ = np.linalg.lstsq(phi, t, rcond = None) #this func returns 4 values, rank,residuals, singular values and weights, we pick only weights
    return w

#gradient descent

def loss_alpha(q, alpha, t):
    return sse(q, alpha, t)

#here we directly dont use the original polynomial for gd, we decompose phi into Q*R, and let alpha <- R*w, and we do gradient descnet on alpha
def batch_gd(q, t, eta, epochs):
    """alpha <- alpha - eta * Q^T (Q alpha - t): update rule per epoch"""
    alpha = np.zeros(q.shape[1])
    hist = [loss_alpha(q, alpha, t)]

    for _ in range(epochs):
        alpha = alpha - eta * (q.T @ (q@alpha - t))
        hist.append(loss_alpha(q, alpha, t))

    return alpha, np.array(hist)


def stochastic_gd(q, t, eta, epochs, seed=SEED):

    '''alpha <- alpha - eta * (q_n^T alpha - t_n) q_n <- update rule after shuffling randomly'''

    N = q.shape[0]
    rng = np.random.default_rng(seed)
    alpha = np.zeros(q.shape[1])
    hist = [loss_alpha(q, alpha, t)]

    for _ in range(epochs):
        order = rng.permutation(N)
        for n in order:
            q_n = q[n]
            alpha = alpha - eta * ((q_n@alpha - t[n]) * q_n)
        hist.append(loss_alpha(q, alpha, t))
    return alpha, np.array(hist)


def minibatch_gd(q, t, eta, epochs, batch = batch_size, seed=SEED):

    ''' now we shuffle and divide our data into mini batches, say 3,3,3,1 and we update the average gradient'''

    rng = np.random.default_rng(seed)
    N = q.shape[0]
    alpha = np.zeros(q.shape[1])
    hist = [loss_alpha(q, alpha, t)]

    for _ in range(epochs):
        order = rng.permutation(N)
        for start in range(0, N, batch):
            indx = order[start:start + batch]
            qb, tb = q[indx], t[indx]
            grad = qb.T @ (qb @ alpha - tb) / len(indx)
            alpha = alpha - eta*grad
        hist.append(loss_alpha(q, alpha, t))
    return alpha, np.array(hist)


def to_w(R, alpha):
    w = np.linalg.solve(R, alpha)
    return w



def run(out_dir = OUTPUT_DIR/"P1"):

    rep = Report(out_dir, "prob:1, polynomial regression : closed form vs family of gradient_descent")
    x,t = load_data("poly_train.csv")
    N = x.size
    rep.line(f"Training set: poly_train.csv, N = {N}")

    #deliverable-1 : fitting polynomials of degree = 1 to 9 and plotting curves against training points

    rep.section("Deliverable 1: closed-form fits, d = 1..9")
    xg = dense_grid(400)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    w_closed = {}

    for d, ax in zip(range(1, D_MAX + 1), axes.ravel()):
        Phi = design_matrix(x, d)
        w = closed_form(Phi, t)
        w_closed[d] = w
        E = sse(Phi, w, t)
        rep.line(f"d = {d}: E(w) = {E:.6e}, cond(Phi) = {np.linalg.cond(Phi):.3e}")
        rep.array(f"   w (d={d})", w)
        ax.plot(xg, true_fun(xg), "g--", lw=1, label=r"$\sin(2\pi x)$ (reference)")
        ax.plot(xg, design_matrix(xg, d) @ w, "r-", lw=1.8, label="closed-form fit")
        ax.scatter(x, t, s=30, facecolors="none", edgecolors="b", label="training data")
        ax.set_title(f"d = {d},  E = {E:.3e}")
        ax.set_ylim(-1.6, 1.6)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("P1.1  Closed-form polynomial fits (N = 10)")
    save_fig(fig, out_dir, "p1_1_closed_form_fits.png")

    # for d = 9

    d = D_MAX
    phi = design_matrix(x, d)
    wc = w_closed[d]
    Q, R = np.linalg.qr(phi, mode="reduced")   # thin QR, Phi = Q R
    rep.section("Setup for d = 9 gradient descent (thin QR, alpha = R w)")
    rep.line(f"cond(Phi) = {np.linalg.cond(phi):.4e}")
    rep.line(f"cond(Q)   = {np.linalg.cond(Q):.6f}")
    rep.line(f"Q shape = {Q.shape};  ||Q^T Q - I||_F = {np.linalg.norm(Q.T @ Q - np.eye(d + 1)):.3e};"
                 f"  ||Q Q^T - I||_F = {np.linalg.norm(Q @ Q.T - np.eye(N)):.3e}")
    rep.line(f"Row norms of Q: {np.array2string(np.linalg.norm(Q, axis=1), precision=6)}")
    rep.line("Starting point alpha = 0 for every variant.")
    rep.line(f"Closed-form E(w_closed) (d=9) = {sse(phi, wc, t):.6e}")

    # deliverable 2 : implementing family of grad_desc for d = 9

    rep.section(f"Deliverables 2-3: Batch / SGD / Mini-batch (size {batch_size}) GD, "
                    f"eta = {eta_main}, {epochs} epochs")
    rep.line("Updates per epoch: Batch = 1, SGD = N = 10, Mini-batch = 4 (batches 3,3,3,1).")

    a_b, h_b = batch_gd(Q, t, eta_main, epochs) #since we already implemented them previously, now we just call them.
    a_s, h_s = stochastic_gd(Q, t, eta_main, epochs, SEED)
    a_mb, h_mb = minibatch_gd(Q, t, eta_main, epochs, batch_size, SEED)

    runs = {"Batch GD": (a_b, h_b), "Stochastic GD": (a_s, h_s),
                "Mini-batch GD (for b=3)": (a_mb, h_mb)}

    for name, (_, h) in runs.items():
        checkpoints = [0, 10, 50, 100, 200, 500, 1000, 1500, epochs]
        vals = ", ".join(f"ep{k}: {h[k]:.3e}" for k in checkpoints)
        rep.line(f"{name:22s} E at {vals}")
    rep.line(f"max |E_SGD - E_Batch| over all epochs = {np.max(np.abs(h_s - h_b)):.3e}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    epochs_axis = np.arange(epochs + 1)
    ax.semilogy(epochs_axis, h_b, "-", lw=3.0, color="tab:blue", label="Batch GD")
    ax.semilogy(epochs_axis, h_s, "--", lw=1.6, color="tab:orange", label="Stochastic GD")
    ax.semilogy(epochs_axis, h_mb, "-", lw=1.6, color="tab:green", label="Mini-batch GD (b = 3)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$E = \frac{1}{2}\|Q\alpha - t\|^2$  (log scale)")
    ax.set_title(rf"P1.3  Loss curves, d = 9, QR-reparametrised, $\eta$ = {eta_main}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, out_dir, "p1_3_loss_curves.png")

    #deliverable 4 : learning rate study

    rep.section(f"Deliverable 4: learning-rate study, Batch GD, {epoch_study} epochs")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {"too_small": "tab:orange", "well_chosen": "tab:blue", "too_large": "tab:red"}
    for regime, eta in eta_study.items():
        a, h = batch_gd(Q, t, eta, epoch_study)
        ax.semilogy(np.arange(epoch_study + 1), h, color=styles[regime], lw=2,
                    label=rf"{regime}: $\eta$ = {eta}")
        rep.line(f"{regime:12s} eta = {eta:<5}: E(0) = {h[0]:.4e}, E(10) = {h[10]:.4e}, "
                     f"E(50) = {h[50]:.4e}, E({epoch_study}) = {h[-1]:.4e}")
        if regime == "too_large":
            w_div = to_w(R, a)
            rep.line(f"  Divergent run (eta = {eta}) after {epoch_study} epochs:")
            rep.line(f"  E = {h[-1]:.6e}")
            rep.array("  w", w_div)
            rep.line(f"  ||w||_2 = {np.linalg.norm(w_div):.6e}  "
                         f"(closed form ||w||_2 = {np.linalg.norm(wc):.6e})")
            rep.line(f"  ||w - w_closed||_2 = {np.linalg.norm(w_div - wc):.6e}")
            rep.line(f"  E grows by a factor (1 - eta)^2 = {(1 - eta) ** 2:.4f} per epoch: "
                         f"E(100)/E(99) = {h[-1] / h[-2]:.4f}")
    
    # eta = 2 boundary case
    a2, h2 = batch_gd(Q, t, eta_boundary, epoch_study)
    ax.semilogy(np.arange(epoch_study + 1), h2, ":", color="k", lw=1.5,
                    label=rf"boundary: $\eta$ = {eta_boundary}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("E (log scale)")
    ax.set_title(r"P1.4  Learning-rate study, Batch GD (QR-reparametrised, d = 9)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_fig(fig, out_dir, "p1_4_learning_rate_study.png")
    
    alpha_star = Q.T @ t
    a_odd, _ = batch_gd(Q, t, eta_boundary, 1)
    rep.line(f"Boundary eta = 2: E stays at E(0) = {h2[0]:.6e} for all epochs "
                 f"(min/max over {epoch_study} epochs: {h2.min():.6e} / {h2.max():.6e}).")
    rep.line(f"  Since Q^T Q = I, alpha_(k+1) - alpha* = (1 - eta)(alpha_k - alpha*) = -(alpha_k - alpha*):"
                 f" the iterate jumps between alpha = 0 and alpha = 2 alpha*; ||alpha_1 - 2 alpha*|| = "
                 f"{np.linalg.norm(a_odd - 2 * alpha_star):.3e}. It neither converges nor diverges.")

    #deliverable 5 : verification

    rep.section(f"Deliverable 5: verification after {epochs} epochs at eta = {eta_main}")
    rep.array("w_closed", wc)
    rep.line(f"||w_closed||_2 = {np.linalg.norm(wc):.6e}")
    verification = {}
    for name, (a, _) in runs.items():
        w_gd = to_w(R, a)
        abs_err = np.linalg.norm(w_gd - wc)
        rel_err = abs_err / np.linalg.norm(wc)
        verification[name] = (abs_err, rel_err)
        rep.array(f"w_GD [{name}]", w_gd)
        rep.line(f"{name:22s} ||w_GD - w_closed||_2 = {abs_err:.6e},  relative = {rel_err:.6e}")

    
    rep.save()



if __name__ == "__main__":
    run()

























    





