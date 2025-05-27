import numpy as np
import os
import ctypes
import ctypes.util
from numpy.ctypeslib import ndpointer
from typing import Tuple, Callable

# 1) load C & C++ libraries
LIB_PATH_C = os.path.join(os.path.dirname(__file__), "../../lib/c/lib_c.so")
LIB_PATH_CPP = os.path.join(os.path.dirname(__file__), "../../lib/cpp/lib_cpp.so")
lib_c = ctypes.CDLL(LIB_PATH_C)
lib_cpp = ctypes.CDLL(LIB_PATH_CPP)

# 2) signature for objective_fn
OBJ_FN = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
)

# 3) wrap your C++ batch solver
lib_cpp.compute_l1_norm_basis_c.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
lib_cpp.compute_l1_norm_basis_c.restype = ctypes.POINTER(ctypes.c_double)


def compute_l1_norm_basis_fast(n: int, weights: np.ndarray) -> np.ndarray:
    """Compute the L1 norm basis using a C++ function for efficiency."""
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    out_rows = ctypes.c_int()
    out_cols = ctypes.c_int()
    ptr = lib_cpp.compute_l1_norm_basis_c(
        n, weights, ctypes.byref(out_rows), ctypes.byref(out_cols)
    )
    rows, cols = out_rows.value, out_cols.value
    buffer = np.ctypeslib.as_array(ptr, shape=(rows * cols,))
    matrix = buffer.reshape((rows, cols), order="F")
    return matrix.copy()


# 4) grab pointers to your four C objectives
c_obj = OBJ_FN(("objective", lib_c))
c_obj_max = OBJ_FN(("objective_max", lib_c))
c_obj_min = OBJ_FN(("objective_min", lib_c))
c_obj_harmonic_mean = OBJ_FN(("objective_harmonic_mean", lib_c))
c_obj_sym = OBJ_FN(("objective_sym", lib_c))
c_obj_denom_sum = OBJ_FN(("objective2", lib_c))
c_obj_denom_max = OBJ_FN(("objective3", lib_c))
c_obj_denom_min = OBJ_FN(("objective4", lib_c))


# 5) declare arg_max_greedy
lib_c.arg_max_greedy.argtypes = [
    ctypes.c_int,  # n_clusters
    ctypes.c_int,  # n_vertices
    ctypes.POINTER(ctypes.c_ubyte),  # tau array ptr
    ctypes.POINTER(ctypes.c_double),  # memo array ptr
    OBJ_FN,  # your callback
]
lib_c.arg_max_greedy.restype = ctypes.POINTER(ctypes.c_int)

# 6) hook up free()
libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None


# 7) generic helper
def _arg_max_greedy_with(
    n_clusters: int, tau: np.ndarray, memo: np.ndarray, obj_cb  # shape = (k, N)
) -> Tuple[int, int]:
    # -> ensure tau is exactly 0/1 bytes, contiguous
    tau8 = np.ascontiguousarray(tau, dtype=np.uint8)
    memo64 = np.ascontiguousarray(memo, dtype=np.float64)
    ptr = lib_c.arg_max_greedy(
        ctypes.c_int(tau8.shape[0]),  # k = current # clusters
        ctypes.c_int(tau8.shape[1]),  # N = # original vertices
        tau8.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        memo64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        obj_cb,
    )
    i, j = ptr[0], ptr[1]
    libc.free(ptr)
    return i, j


# 8) Python‐friendly wrappers
def arg_max_greedy_undirected(
    n: int,
    tau: np.ndarray,
    memo: np.ndarray,
) -> Tuple[int, int]:
    return _arg_max_greedy_with(n, tau, memo, c_obj)


def arg_max_greedy_denom_sum(
    n: int,
    tau: np.ndarray,
    memo: np.ndarray,
) -> Tuple[int, int]:
    return _arg_max_greedy_with(n, tau, memo, c_obj_denom_sum)


def arg_max_greedy_denom_max(
    n: int,
    tau: np.ndarray,
    memo: np.ndarray,
) -> Tuple[int, int]:
    return _arg_max_greedy_with(n, tau, memo, c_obj_denom_max)


def arg_max_greedy_denom_min(
    n: int,
    tau: np.ndarray,
    memo: np.ndarray,
) -> Tuple[int, int]:
    return _arg_max_greedy_with(n, tau, memo, c_obj_denom_min)


def arg_max_greedy_max(
    n: int,
    tau: np.ndarray,
    memo: np.ndarray,
) -> Tuple[int, int]:
    return _arg_max_greedy_with(n, tau, memo, c_obj_max)


def arg_max_greedy_min(
    n: int,
    tau: np.ndarray,
    memo: np.ndarray,
) -> Tuple[int, int]:
    return _arg_max_greedy_with(n, tau, memo, c_obj_min)


def arg_max_greedy_sym(
    n: int,
    tau: np.ndarray,
    memo: np.ndarray,
) -> Tuple[int, int]:
    return _arg_max_greedy_with(n, tau, memo, c_obj_sym)


def arg_max_greedy_harmonic_mean(
    n: int,
    tau: np.ndarray,
    memo: np.ndarray,
) -> Tuple[int, int]:
    return _arg_max_greedy_with(n, tau, memo, c_obj_harmonic_mean)


# 9) Some useful aliases
__objective_t__ = Callable[[int, np.ndarray, np.ndarray], float]

__xrank_fn_undirected__ = {
    "W(A, B) / (|A| * |B|)": arg_max_greedy_undirected,
    "W(A, B) / (|A| + |B|)": arg_max_greedy_denom_sum,
    "W(A, B) / max(|A|, |B|)": arg_max_greedy_denom_max,
    "W(A, B) / min(|A|, |B|)": arg_max_greedy_denom_min,
}

__xrank_fn_directed__ = {
    "Max": arg_max_greedy_max,
    "Min": arg_max_greedy_min,
    "Symmetric": arg_max_greedy_sym,
    "Harmonic Mean": arg_max_greedy_harmonic_mean,
}
