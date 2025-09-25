import time
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from typing import Literal, Tuple, Optional

# pylspack kernels
# - rmcgs: dense (row-major) CountSketch and/or Gaussian (G*S*A)
# - csrcgs: CSR version (S or G*S)
# - csrjlt / rmdsc: helpers (Gaussian-only JL transform for CSR; diagonal scaling)
from pylspack.linalg_kernels import rmcgs, csrcgs, csrjlt

# ------------------------------
# Utilities
# ------------------------------

def _as_row_major(A: np.ndarray) -> np.ndarray:
    """Return a C_CONTIGUOUS (row-major) view/copy of A."""
    return np.ascontiguousarray(A, dtype=np.float64)

def lipschitz_norm_xtx(X: np.ndarray, power_iters: int = 20) -> float:
    """Estimate spectral norm of X^T X via power iteration on X (works for dense or CSR)."""
    n, d = X.shape
    v = np.random.randn(d)
    v /= norm(v) + 1e-12
    for _ in range(power_iters):
        if sparse.issparse(X):
            z = X @ v
            v = X.T @ z
        else:
            z = X.dot(v)
            v = X.T.dot(z)
        nv = norm(v) + 1e-12
        v /= nv
    # Rayleigh quotient on X^T X
    if sparse.issparse(X):
        z = X @ v
    else:
        z = X.dot(v)
    return float(np.dot(z, z))  # = ||X v||_2^2

def prox_l1(x: np.ndarray, tau: float) -> np.ndarray:
    """Soft-thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

def objective_value(X, y, beta, reg: Literal["lasso","ridge"], lam: float) -> float:
    r = X @ beta - y
    data = 0.5 * float(np.dot(r, r))
    if reg == "lasso":
        return data + lam * float(np.sum(np.abs(beta)))
    else:
        return data + 0.5 * lam * float(np.dot(beta, beta))

# ------------------------------
# Sketching with pylspack
# ------------------------------

def make_sketch(X, m: int, kind: Literal["countsketch","gaussian"]) -> np.ndarray:
    """
    Build X_tilde = P X using pylspack:
      - 'countsketch' : CountSketch rows (sparse subspace embedding)
      - 'gaussian'    : Gaussian subspace embedding
    m = number of sketched rows.
    """
    if sparse.issparse(X):
        if kind == "countsketch":
            # csrcgs(A, m, r): returns G S A if r>0 else S A. We want S A.
            X_tilde = csrcgs(X.tocsr(), m, 0)
        else:
            # Gaussian-only JL on CSR
            X_tilde = csrjlt(X.tocsr(), m)
    else:
        A = _as_row_major(np.asarray(X))
        if kind == "countsketch":
            # rmcgs(A, m, r): G S A if r>0 else S A. We want S A.
            X_tilde = rmcgs(A, m, 0)
        else:
            # Gaussian-only: rmcgs with CountSketch size r=0 then multiply by Gaussian of size m
            # pylspack exposes csrjlt only for CSR. For dense, route through a CSR wrapper if needed.
            # Simpler: convert to CSR and use csrjlt for Gaussian for correctness.
            X_csr = sparse.csr_matrix(A)
            X_tilde = csrjlt(X_csr, m).toarray()
    return X_tilde

# ------------------------------
# Iterative SRO
# ------------------------------

def iterative_sro(
    X, y,
    lam: float,
    reg: Literal["lasso","ridge"] = "lasso",
    X_tilde: Optional[np.ndarray] = None,
    N: int = 10,
    step_scale: float = 0.9,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Iterative SRO (Algorithm 1):
      beta^{t} = argmin_beta 0.5||X_tilde (beta - beta^{t-1})||_2^2 - <y - X beta^{t-1}, X beta> + h_lambda(beta)
    We solve each sketched subproblem with proximal gradient (lasso) or gradient step (ridge).
    """
    n, d = X.shape
    if X_tilde is None:
        raise ValueError("Provide X_tilde = P X (sketch once, reuse across iterations).")

    # Precompute for fast gradient: we only need X^T(y - X beta^{t-1}) and X_tilde^T X_tilde
    if sparse.issparse(X_tilde):
        XtX_tilde = (X_tilde.T @ X_tilde)
    else:
        XtX_tilde = X_tilde.T.dot(X_tilde)

    # Lipschitz for the sketched quadratic term
    L = lipschitz_norm_xtx(X_tilde)
    eta = step_scale / (L + 1e-12)

    beta = np.zeros(d, dtype=np.float64)
    hist = {"obj": [], "time": []}

    t0 = time.time()
    for t in range(1, N+1):
        # Gradient of the sketched quadratic at current beta:
        # grad_q(beta) = XtX_tilde @ (beta - beta_prev) - X^T(y - X beta_prev) evaluated at beta (see Algorithm 1 form)
        # A more convenient equivalent (deriving (5) in the paper) is a single gradient on:
        #  0.5||X_tilde (beta - beta_prev)||^2 - <y - X beta_prev, X beta>
        # => grad = XtX_tilde @ (beta - beta_prev) - X^T(y - X beta_prev)
        # We maintain beta_prev as last iterate
        # Implement a few proximal gradient steps per outer-iteration (1 is often enough in practice).
        beta_prev = beta.copy()

        # Compute b_term = X^T(y - X beta_prev)
        r_prev = y - (X @ beta_prev if not sparse.issparse(X) else X.dot(beta_prev))
        b_term = (X.T @ r_prev) if sparse.issparse(X) else X.T.dot(r_prev)

        # One proximal gradient step
        grad = XtX_tilde @ (beta - beta_prev) - b_term
        z = beta - eta * grad

        if reg == "lasso":
            beta = prox_l1(z, eta * lam)
        else:
            # Ridge: proximal is shrinkage
            beta = z / (1.0 + eta * lam)

        if verbose and (t % 5 == 0 or t == 1 or t == N):
            obj = objective_value(X, y, beta, reg, lam)
            hist["obj"].append(obj)
            hist["time"].append(time.time() - t0)

    return beta, hist

# ------------------------------
# Reference solver (unsketched) via proximal gradient
# ------------------------------

def pgd_reference(X, y, lam: float, reg: Literal["lasso","ridge"]="lasso",
                  steps: int = 200, step_scale: float=0.9) -> np.ndarray:
    n, d = X.shape
    L = lipschitz_norm_xtx(X)
    eta = step_scale / (L + 1e-12)
    beta = np.zeros(d)

    for _ in range(steps):
        r = y - (X @ beta if not sparse.issparse(X) else X.dot(beta))
        grad = -(X.T @ r) if sparse.issparse(X) else -X.T.dot(r)
        z = beta - eta * grad
        if reg == "lasso":
            beta = prox_l1(z, eta * lam)
        else:
            beta = z / (1.0 + eta * lam)
    return beta

# ------------------------------
# Benchmark runner
# ------------------------------

def run_demo(seed=0):
    rng = np.random.default_rng(seed)
    n, d, s = 20000, 1000, 50  # tall-and-thin; sparsity for Lasso test
    # Synthetic sparse-signal regression
    X = rng.standard_normal((n, d))
    beta_true = np.zeros(d)
    supp = rng.choice(d, size=s, replace=False)
    beta_true[supp] = rng.standard_normal(s)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    # Regularizer
    reg = "lasso"
    lam = 0.01 * np.sqrt(2.0 * np.log(d) / n)  # order from theory

    # Reference (no sketch)
    t0 = time.time()
    beta_ref = pgd_reference(X, y, lam, reg, steps=300)
    t_ref = time.time() - t0

    # Choose sketch size m (rows)
    # For Gaussian/CountSketch embeddings, m ~ O(r log(1/delta)/epsilon^2).
    # Heuristic: a small multiple of d's numerical rank; start with ~4*d_sparsity.
    m = 8 * s

    results = {}
    for kind in ["countsketch", "gaussian"]:
        t1 = time.time()
        X_tilde = make_sketch(X, m=m, kind=kind)
        t_sk = time.time() - t1

        t2 = time.time()
        beta_sro, hist = iterative_sro(X, y, lam, reg=reg, X_tilde=X_tilde, N=12, verbose=False)
        t_sro = time.time() - t2

        # Metrics
        rel_x_err = norm(X @ (beta_sro - beta_ref)) / (norm(X @ beta_ref) + 1e-12)
        obj = objective_value(X, y, beta_sro, reg, lam)

        results[kind] = {
            "m": m,
            "time_sketch_s": t_sk,
            "time_iter_sro_s": t_sro,
            "time_ref_pg_s": t_ref,
            "rel_X_norm_err": float(rel_x_err),
            "objective": float(obj),
        }

    return results

if __name__ == "__main__":
    out = run_demo()
    print("== Iterative SRO (pylspack) comparison ==")
    for k, v in out.items():
        print(f"[{k}] m={v['m']}, rel_X_err={v['rel_X_norm_err']:.4f}, "
              f"t_sketch={v['time_sketch_s']:.2f}s, t_iter={v['time_iter_sro_s']:.2f}s, t_ref={v['time_ref_pg_s']:.2f}s")
