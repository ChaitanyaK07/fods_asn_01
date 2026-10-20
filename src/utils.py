
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

SEED = 42 


def true_fun(x):
    return np.sin(2.0*np.pi*x)

def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the provided CSV files inside {DATA_DIR}."
        )
    arr = np.loadtxt(path, delimiter = ',', skiprows = 1, ndmin=2)
    return arr[:, 0].astype(float), arr[:, 1].astype(float)




def design_matrix(x, d):

    # here design matrix (phi) is of dim N*(d+1) with n-th row {1, x_n, x_n^2.....x_n^d}

    x = np.asarray(x, dtype= float)
    phi = np.ones((x.size,  d + 1))

    for j in range(1, d+1):
        phi[:,j] = phi[:,j-1] * x
    return phi


'''def legendre_design(x, d):
    """Phi_{nj} = P_j(2 x_n - 1), j = 0..d, built from the recurrence
    (j+1) P_{j+1}(u) = (2j+1) u P_j(u) - j P_{j-1}(u),  P_0 = 1, P_1 = u.
    """
    u = 2.0 * np.asarray(x, dtype=float) - 1.0
    Phi = np.zeros((u.size, d + 1))
    Phi[:, 0] = 1.0
    if d >= 1:
        Phi[:, 1] = u
    for j in range(1, d):
        Phi[:, j + 1] = ((2 * j + 1) * u * Phi[:, j] - j * Phi[:, j - 1]) / (j + 1)
    return Phi '''



#metrics

def sse(phi, w, t):

    r = phi@w - t
    return 0.5 * float(r@r)

def rmse(phi, w, t):

    r = phi@w - t
    return float(np.sqrt(np.mean(r**2)))



#for report writing

class Report:
    """writes them to <out_dir>/results.txt"""

    def __init__(self, out_dir, title):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lines = []
        self.section(title, char="=")

    def line(self, text=""):
        print(text)
        self.lines.append(str(text))

    def section(self, title, char="-"):
        self.line()
        self.line(title)
        self.line(char * len(title))

    def array(self, name, arr, fmt="{: .6e}"):
        arr = np.atleast_1d(arr)
        body = ", ".join(fmt.format(v) for v in arr)
        self.line(f"{name} = [{body}]")

    def save(self, filename="results.txt"):
        path = self.out_dir / filename
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return path


def save_fig(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def dense_grid(n=200):
    """Dense evenly spaced grid on [0, 1] for plotting curves."""
    return np.linspace(0.0, 1.0, n)









        
 


    






