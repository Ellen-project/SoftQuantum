from __future__ import annotations
import math
import numpy as np
from typing import Iterable, List, Sequence, Tuple, Optional

# Optional CUDA backend
try:
    import _svcuda as svcuda  # built from cuda_statevector.cu via pybind11
    _HAVE_CUDA = True
except Exception:
    svcuda = None
    _HAVE_CUDA = False

Array = np.ndarray
_EPS = 1e-12

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _validate_qubits(n_qubits: int, qs: Sequence[int]):
    for q in qs:
        if not (0 <= q < n_qubits):
            raise IndexError(f"Qubit index {q} out of range for {n_qubits} qubits")


def _as_tuple(x: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in x)


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------
class QuantumSimulator:
    def __init__(self, num_qubits: int, seed: Optional[int] = None):
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = int(num_qubits)
        self.dim = 1 << self.num_qubits
        self.dtype = np.complex128  # fixed double precision
        self.rng = np.random.default_rng(seed)
        # |0...0> statevector
        self.state: Array = np.zeros(self.dim, dtype=self.dtype)
        self.state[0] = 1.0 + 0.0j
        # classical bits (optional)
        self.creg: List[int] = [0] * self.num_qubits

    # ----------------------------- Numerics ---------------------------------
    def _axis_of(self, q: int) -> int:
        # LSB-on-the-right convention: qubit 0 corresponds to the last axis
        return self.num_qubits - 1 - int(q)

    def _as_tensor(self) -> Array:
        return self.state.reshape((2,) * self.num_qubits)

    def _normalize_(self):
        norm2 = float(np.vdot(self.state, self.state).real)
        if abs(norm2 - 1.0) > 1e-15 and norm2 > _EPS:
            self.state /= math.sqrt(norm2)
        # prune tiny numerical noise
        self.state.real[np.abs(self.state.real) < _EPS] = 0.0
        self.state.imag[np.abs(self.state.imag) < _EPS] = 0.0

    # -------------------------- Unitary application -------------------------
    def apply_unitary(self, targets: Sequence[int], U: Array):
        """Apply arbitrary k-qubit unitary U to the given targets (global)."""
        t = _as_tuple(targets)
        k = len(t)
        if k == 0:
            return
        _validate_qubits(self.num_qubits, t)
        m = 1 << k
        U = np.asarray(U, dtype=self.dtype)
        if U.shape != (m, m):
            raise ValueError(f"U must be {(m, m)} for k={k}, got {U.shape}")

        # CUDA fast path for k in {1,2} and generic k (3..16)
        if _HAVE_CUDA:
            try:
                if k == 1:
                    svcuda.apply_1q(self.state, self.num_qubits, int(t[0]), U)
                    self._normalize_()
                    return
                elif k == 2:
                    svcuda.apply_2q(self.state, self.num_qubits, int(t[0]), int(t[1]), U)
                    self._normalize_()
                    return
                elif 3 <= k <= 16:
                    # Memory note: full U is uploaded (16 * 4^k bytes for complex128).
                    svcuda.apply_kq(self.state, self.num_qubits, np.array(t, dtype=np.int32), U)
                    self._normalize_()
                    return
            except Exception:
                # fall back to CPU
                pass

        # CPU vectorized path
        psi = self._as_tensor()
        axes = tuple(self._axis_of(q) for q in t)
        rest_axes = tuple(ax for ax in range(self.num_qubits) if ax not in axes)
        moved = np.moveaxis(psi, axes + rest_axes, tuple(range(self.num_qubits)))
        front = moved.reshape((m, -1))
        front2 = U @ front
        moved2 = front2.reshape((2,) * self.num_qubits)
        # Move axes back
        inv_perm = np.argsort((*axes, *rest_axes))
        psi2 = np.moveaxis(moved2, tuple(range(self.num_qubits)), (*axes, *rest_axes))
        psi2 = psi2.transpose(inv_perm)
        self.state = psi2.reshape(self.dim).astype(self.dtype, copy=False)
        self._normalize_()

    def apply_controlled_unitary(
        self,
        controls: Sequence[int],
        targets: Sequence[int],
        U: Array,
        ctrl_state: Optional[Sequence[int]] = None,
    ):
        """Apply U to targets conditioned on control bits matching ctrl_state.
        By default, ctrl_state is all 1s. Supports any #controls and any k."""
        c = _as_tuple(controls)
        t = _as_tuple(targets)
        _validate_qubits(self.num_qubits, (*c, *t))
        if set(c) & set(t):
            raise ValueError("controls and targets must be disjoint")
        if ctrl_state is None:
            ctrl_state = (1,) * len(c)
        ctrl_state = _as_tuple(ctrl_state)
        if len(ctrl_state) != len(c):
            raise ValueError("ctrl_state length must match number of controls")

        k = len(t)
        m = 1 << k
        U = np.asarray(U, dtype=self.dtype)
        if U.shape != (m, m):
            raise ValueError(f"U must be {(m, m)} for k={k}, got {U.shape}")

        # CUDA fast path for 1q/2q targets with all-ones control state
        if _HAVE_CUDA and (k in (1, 2)) and all(b == 1 for b in ctrl_state):
            try:
                # Build control bitmask (1 where control must be 1)
                mask = 0
                for q in c:
                    mask |= (1 << q)
                if k == 1:
                    svcuda.apply_c1q(self.state, self.num_qubits, mask, int(t[0]), U)
                else:
                    svcuda.apply_c2q(self.state, self.num_qubits, mask, int(t[0]), int(t[1]), U)
                self._normalize_()
                return
            except Exception:
                pass

        # CPU path
        psi = self._as_tensor()
        ax_c = tuple(self._axis_of(q) for q in c)
        ax_t = tuple(self._axis_of(q) for q in t)
        rest = tuple(ax for ax in range(self.num_qubits) if ax not in (*ax_c, *ax_t))
        moved = np.moveaxis(psi, (*ax_c, *ax_t, *rest), tuple(range(self.num_qubits)))
        idx_ctrl = tuple(int(b) for b in ctrl_state)
        slices = idx_ctrl + (slice(None),) * (len(ax_t) + len(rest))
        block = moved[slices]
        front = block.reshape((m, -1))
        front2 = U @ front
        moved[slices] = front2.reshape(block.shape)
        inv_perm = np.argsort((*ax_c, *ax_t, *rest))
        out = np.moveaxis(moved, tuple(range(self.num_qubits)), (*ax_c, *ax_t, *rest))
        out = out.transpose(inv_perm)
        self.state = out.reshape(self.dim).astype(self.dtype, copy=False)
        self._normalize_()

    def apply_global_unitary_full(self, U_full: Array):
        """Apply a full 2^n x 2^n unitary. Only feasible for small n."""
        U_full = np.asarray(U_full, dtype=self.dtype)
        if U_full.shape != (self.dim, self.dim):
            raise ValueError(f"U_full must be {(self.dim, self.dim)}")
        self.state = (U_full @ self.state).astype(self.dtype, copy=False)
        self._normalize_()

    # ------------------------------ Noise -----------------------------------
    def _apply_operator(self, targets: Sequence[int], M: Array) -> Array:
        t = _as_tuple(targets)
        k = len(t)
        if k == 0:
            return self.state.copy()
        _validate_qubits(self.num_qubits, t)
        m = 1 << k
        M = np.asarray(M, dtype=self.dtype)
        if M.shape != (m, m):
            raise ValueError(f"Operator must be {(m, m)} for k={k}")
        psi = self._as_tensor()
        axes = tuple(self._axis_of(q) for q in t)
        rest_axes = tuple(ax for ax in range(self.num_qubits) if ax not in axes)
        moved = np.moveaxis(psi, axes + rest_axes, tuple(range(self.num_qubits)))
        front = moved.reshape((m, -1))
        front2 = M @ front
        moved2 = front2.reshape((2,) * self.num_qubits)
        inv_perm = np.argsort((*axes, *rest_axes))
        psi2 = np.moveaxis(moved2, tuple(range(self.num_qubits)), (*axes, *rest_axes))
        psi2 = psi2.transpose(inv_perm)
        return psi2.reshape(self.dim).astype(self.dtype, copy=False)

    def apply_channel(self, targets: Sequence[int], kraus_ops: Sequence[Array]):
        ops = [np.asarray(K, dtype=self.dtype) for K in kraus_ops]
        candidates: List[Array] = []
        probs: List[float] = []
        for K in ops:
            v = self._apply_operator(targets, K)
            p = float(np.vdot(v, v).real)
            candidates.append(v)
            probs.append(max(p, 0.0))
        s = sum(probs)
        if s < _EPS:
            return
        probs = [p / s for p in probs]
        idx = int(self.rng.choice(len(ops), p=probs))
        v = candidates[idx]
        self.state = v / math.sqrt(max(np.vdot(v, v).real, _EPS))

    # ---------------------------- Standard gates ----------------------------
    # Base single-qubit rotations
    @staticmethod
    def Ux(theta: float) -> Array:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def Uy(theta: float) -> Array:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def Uz(theta: float) -> Array:
        a = theta / 2
        return np.array([[np.exp(-1j * a), 0], [0, np.exp(1j * a)]], dtype=np.complex128)

    # General 1-qubit U3(θ, φ, λ)
    @staticmethod
    def U3(theta: float, phi: float, lam: float) -> Array:
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
            dtype=np.complex128,
        )

    # Common 1q gates
    def I(self, q: int):
        self.apply_unitary([q], np.eye(2, dtype=self.dtype))

    def H(self, q: int):
        inv = 1.0 / math.sqrt(2.0)
        self.apply_unitary([q], np.array([[inv, inv], [inv, -inv]], dtype=self.dtype))

    def S(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, 1j]], dtype=self.dtype))

    def Sdg(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, -1j]], dtype=self.dtype))

    def T(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=self.dtype))

    def Tdg(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=self.dtype))

    def X(self, q: int):
        self.apply_unitary([q], np.array([[0, 1], [1, 0]], dtype=self.dtype))

    def Y(self, q: int):
        self.apply_unitary([q], np.array([[0, -1j], [1j, 0]], dtype=self.dtype))

    def Z(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, -1]], dtype=self.dtype))

    def RX(self, q: int, theta: float):
        self.apply_unitary([q], self.Ux(theta))

    def RY(self, q: int, theta: float):
        self.apply_unitary([q], self.Uy(theta))

    def RZ(self, q: int, theta: float):
        self.apply_unitary([q], self.Uz(theta))

    def U(self, q: int, theta: float, phi: float, lam: float):
        self.apply_unitary([q], self.U3(theta, phi, lam))

    # Two-qubit gates
    def SWAP(self, q1: int, q2: int):
        U = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype)
        self.apply_unitary([q1, q2], U)

    def ISWAP(self, q1: int, q2: int):
        U = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=self.dtype)
        self.apply_unitary([q1, q2], U)

    def ISWAP_theta(self, q1: int, q2: int, theta: float):
        """Generalized iSWAP with rotation angle theta (theta=pi/2 -> iSWAP)."""
        c, s = math.cos(theta), math.sin(theta)
        U = np.array(
            [[1, 0, 0, 0],
             [0, c, 1j * s, 0],
             [0, 1j * s, c, 0],
             [0, 0, 0, 1]],
            dtype=self.dtype
        )
        self.apply_unitary([q1, q2], U)

    def ISWAP_pow(self, q1: int, q2: int, p: float):
        """iSWAP**p: equivalent to ISWAP_theta(theta=p*pi/2)."""
        self.ISWAP_theta(q1, q2, p * math.pi / 2.0)

    def ISWAPdg(self, q1: int, q2: int):
        self.ISWAP_theta(q1, q2, -math.pi / 2.0)

    def fSim(self, q1: int, q2: int, theta: float, phi: float):
        c, s = math.cos(theta), math.sin(theta)
        U = np.array(
            [[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]],
            dtype=self.dtype,
        )
        self.apply_unitary([q1, q2], U)

    def SYC(self, q1: int, q2: int):
        """Sycamore gate = FSim(pi/2, pi/6)."""
        self.fSim(q1, q2, math.pi / 2.0, math.pi / 6.0)

    def RXX(self, q1: int, q2: int, theta: float):
        a = theta / 2
        I4 = np.eye(4, dtype=self.dtype)
        XX = np.kron(np.array([[0, 1], [1, 0]], dtype=self.dtype), np.array([[0, 1], [1, 0]], dtype=self.dtype))
        self.apply_unitary([q1, q2], (math.cos(a) * I4) - 1j * math.sin(a) * XX)

    def RYY(self, q1: int, q2: int, theta: float):
        a = theta / 2
        I4 = np.eye(4, dtype=self.dtype)
        Y = np.array([[0, -1j], [1j, 0]], dtype=self.dtype)
        YY = np.kron(Y, Y)
        self.apply_unitary([q1, q2], (math.cos(a) * I4) - 1j * math.sin(a) * YY)

    def RZZ(self, q1: int, q2: int, theta: float):
        a = theta / 2
        e = np.exp(-1j * a)
        ed = np.exp(1j * a)
        U = np.diag([e, ed, ed, e]).astype(self.dtype)
        self.apply_unitary([q1, q2], U)

    # PhasedFSim (generalized excitation-preserving two-qubit gate)
    @staticmethod
    def _PhasedFSim_matrix(theta: float, zeta: float = 0.0, chi: float = 0.0, gamma: float = 0.0, phi: float = 0.0) -> Array:
        # Matches Cirq doc: U = [[1,0,0,0],
        # [0, e^{-i(γ+ζ)}cosθ, -i e^{-i(γ-χ)} sinθ, 0],
        # [0, -i e^{-i(γ+χ)} sinθ, e^{-i(γ-ζ)} cosθ, 0],
        # [0, 0, 0, e^{-i(2γ+φ)}]]
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.exp(-1j*(gamma+zeta)) * c, -1j * np.exp(-1j*(gamma-chi)) * s, 0.0],
            [0.0, -1j * np.exp(-1j*(gamma+chi)) * s, np.exp(-1j*(gamma - zeta)) * c, 0.0],
            [0.0, 0.0, 0.0, np.exp(-1j*(2*gamma + phi))]
        ], dtype=np.complex128)

    def PhasedFSim(self, q1: int, q2: int, theta: float, zeta: float = 0.0, chi: float = 0.0, gamma: float = 0.0, phi: float = 0.0):
        U = self._PhasedFSim_matrix(theta, zeta, chi, gamma, phi)
        self.apply_unitary([q1, q2], U)

    def CZ_wave(self, q1: int, q2: int, phi: float, zeta: float = 0.0, chi: float = 0.0, gamma: float = 0.0):
        """CZ with waveform-like parameterization = PhasedFSim(θ=0, φ=phi, ζ, χ, γ)."""
        self.PhasedFSim(q1, q2, 0.0, zeta=zeta, chi=chi, gamma=gamma, phi=phi)

    def PhasedISWAP(self, q1: int, q2: int, theta: float, phase: float):
        """Phased iSWAP: (Rz(φ/2)⊗Rz(-φ/2)) · ISWAP_theta(θ) · (Rz(-φ/2)⊗Rz(φ/2))."""
        Rz = lambda a: np.array([[np.exp(-1j*a/2), 0],[0, np.exp(1j*a/2)]], dtype=self.dtype)
        pre  = np.kron(Rz(phase/2), Rz(-phase/2))
        post = np.kron(Rz(-phase/2), Rz(phase/2))
        c, s = math.cos(theta), math.sin(theta)
        core = np.array(
            [[1, 0, 0, 0],
             [0, c, 1j * s, 0],
             [0, 1j * s, c, 0],
             [0, 0, 0, 1]],
            dtype=self.dtype
        )
        U = pre @ core @ post
        self.apply_unitary([q1, q2], U)

    # Controlled gates
    def CX(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[0, 1], [1, 0]], dtype=self.dtype))

    def CY(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[0, -1j], [1j, 0]], dtype=self.dtype))

    def CZ(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[1, 0], [0, -1]], dtype=self.dtype))

    def CH(self, c: int, t: int):
        inv = 1.0 / math.sqrt(2.0)
        self.apply_controlled_unitary([c], [t], np.array([[inv, inv], [inv, -inv]], dtype=self.dtype))

    def CS(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[1, 0], [0, 1j]], dtype=self.dtype))

    def CT(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=self.dtype))

    def CP(self, c: int, t: int, lam: float):
        self.apply_controlled_unitary([c], [t], np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=self.dtype))

    def CRX(self, c: int, t: int, theta: float):
        self.apply_controlled_unitary([c], [t], self.Ux(theta))

    def CRY(self, c: int, t: int, theta: float):
        self.apply_controlled_unitary([c], [t], self.Uy(theta))

    def CRZ(self, c: int, t: int, theta: float):
        self.apply_controlled_unitary([c], [t], self.Uz(theta))

    def CU(self, c: int, t: int, U: Array):
        U = np.asarray(U, dtype=self.dtype)
        if U.shape != (2, 2):
            raise ValueError("CU expects a 2x2 unitary for the target qubit")
        self.apply_controlled_unitary([c], [t], U)

    def Toffoli(self, c1: int, c2: int, t: int):
        self.apply_controlled_unitary([c1, c2], [t], np.array([[0, 1], [1, 0]], dtype=self.dtype))

    def CSWAP(self, c: int, q1: int, q2: int):
        U = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype)
        self.apply_controlled_unitary([c], [q1, q2], U)

    # ------------------------------ Measurement -----------------------------
    def measure(self, q: int, cbit: Optional[int] = None) -> int:
        ax = self._axis_of(q)
        psi = self._as_tensor()
        moved = np.moveaxis(psi, ax, -1)
        amp0 = moved[..., 0].reshape(-1)
        amp1 = moved[..., 1].reshape(-1)
        p0 = float((amp0.conj() * amp0).real.sum())
        p0 = min(max(p0, 0.0), 1.0)
        p1 = max(0.0, 1.0 - p0)
        outcome = int(self.rng.random() < p1)
        if outcome == 0:
            moved[..., 1] = 0.0
        else:
            moved[..., 0] = 0.0
        self.state = moved.reshape(self.dim)
        self._normalize_()
        if cbit is not None and 0 <= cbit < len(self.creg):
            self.creg[cbit] = outcome
        return outcome

    def measure_all(self) -> List[int]:
        outcomes = []
        for q in range(self.num_qubits - 1, -1, -1):
            outcomes.append(self.measure(q))
        return list(reversed(outcomes))

    def reset(self, q: int):
        # Projectively reset to |0>
        ax = self._axis_of(q)
        psi = self._as_tensor()
        moved = np.moveaxis(psi, ax, -1)
        moved[..., 1] = 0.0
        self.state = moved.reshape(self.dim)
        self._normalize_()

    # ----------------------------- Debug helpers ----------------------------
    def probs(self) -> Array:
        p = np.abs(self.state) ** 2
        s = float(p.sum())
        return p / (s if s > _EPS else 1.0)

    def state_ket(self, tol: float = 1e-9) -> str:
        out = []
        p = self.probs()
        for i, (amp, pr) in enumerate(zip(self.state, p)):
            if pr > tol:
                out.append(f"({amp.real:+.6f}{amp.imag:+.6f}j)|{i:0{self.num_qubits}b}>")
        return " + ".join(out) if out else "0"


# ----------------------------------------------------------------------------
# QASM-like parser/executor (extended)
# ----------------------------------------------------------------------------

def _parse_complex_qasm(tok: str) -> complex:
    t = tok.strip()
    if t.endswith('i') and 'j' not in t:
        t = t[:-1] + 'j'
    t = t.strip('()')
    return complex(t)


def _parse_q_list(s: str) -> List[int]:
    # accepts "1,2,3" or single "5"
    return [int(x) for x in s.split(',') if x != '']


def execute_qasm(sim: QuantumSimulator, lines: Optional[List[str]] = None):
    """Extended QASM-like REPL wired to this engine.

    New:
      - u <q1, q2, ...> <m*m complex entries row-major>   (m = 2^k)
      - u_full <N*N complex entries>   (N = 2^n, beware huge!)
      - iswap_theta q1 q2 theta | iswap_pow q1 q2 p | iswapdg q1 q2
      - phased_iswap q1 q2 theta phase
      - fsim q1 q2 theta phi | syc q1 q2
      - phasedfsim q1 q2 theta zeta chi gamma phi
      - cz_wave q1 q2 phi [zeta chi gamma]
    Existing (subset): qreg/creg, x/y/z/h/s/sdg/t/tdg, rx/ry/rz/u3, swap/iswap,
      cx/cy/cz/ch/cs/ct/crx/cry/crz/cp, toffoli/ccx, cswap, rxx/ryy/rzz,
      measure [cbit], noise_* (nbf/npf/ndp/nph), seed, print_*, run, reset.
    """
    if lines is None:
        print("Enter QASM commands; type 'run' to execute:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == 'run':
                break
            lines.append(line)

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('//') or line.startswith('#'):
            continue
        parts = line.replace(';', ' ').split()
        cmd = parts[0].lower()
        try:
            if cmd == 'qreg':
                n = int(parts[1].split('[')[1].split(']')[0]) if '[' in parts[1] else int(parts[1])
                sim.__init__(n)
            elif cmd == 'creg':
                m = int(parts[1].split('[')[1].split(']')[0]) if '[' in parts[1] else int(parts[1])
                sim.creg = [0] * m
            elif cmd in ('i', 'id'):
                sim.I(int(parts[1]))
            elif cmd == 'h':
                sim.H(int(parts[1]))
            elif cmd == 's':
                sim.S(int(parts[1]))
            elif cmd == 'sdg':
                sim.Sdg(int(parts[1]))
            elif cmd == 't':
                sim.T(int(parts[1]))
            elif cmd == 'tdg':
                sim.Tdg(int(parts[1]))
            elif cmd == 'x':
                sim.X(int(parts[1]))
            elif cmd == 'y':
                sim.Y(int(parts[1]))
            elif cmd == 'z':
                sim.Z(int(parts[1]))
            elif cmd == 'rx':
                sim.RX(int(parts[1]), float(parts[2]))
            elif cmd == 'ry':
                sim.RY(int(parts[1]), float(parts[2]))
            elif cmd == 'rz':
                sim.RZ(int(parts[1]), float(parts[2]))
            elif cmd == 'u3':
                q = int(parts[1]); th, ph, la = map(float, parts[2:5])
                sim.U(q, th, ph, la)
            elif cmd == 'swap':
                sim.SWAP(int(parts[1]), int(parts[2]))
            elif cmd == 'iswap':
                sim.ISWAP(int(parts[1]), int(parts[2]))
            elif cmd == 'iswap_theta':
                q1, q2, th = int(parts[1]), int(parts[2]), float(parts[3])
                sim.ISWAP_theta(q1, q2, th)
            elif cmd == 'iswap_pow':
                q1, q2, p = int(parts[1]), int(parts[2]), float(parts[3])
                sim.ISWAP_pow(q1, q2, p)
            elif cmd == 'iswapdg':
                sim.ISWAPdg(int(parts[1]), int(parts[2]))
            elif cmd == 'phased_iswap':
                q1, q2, th, ph = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
                sim.PhasedISWAP(q1, q2, th, ph)
            elif cmd == 'fsim':
                q1, q2 = int(parts[1]), int(parts[2]); th, ph = map(float, parts[3:5])
                sim.fSim(q1, q2, th, ph)
            elif cmd == 'syc':
                sim.SYC(int(parts[1]), int(parts[2]))
            elif cmd == 'phasedfsim':
                q1, q2 = int(parts[1]), int(parts[2])
                th, ze, ch, ga, ph = map(float, parts[3:8])
                sim.PhasedFSim(q1, q2, th, ze, ch, ga, ph)
            elif cmd == 'cz_wave':
                q1, q2 = int(parts[1]), int(parts[2])
                if len(parts) == 4:
                    sim.CZ_wave(q1, q2, float(parts[3]))
                elif len(parts) == 7:
                    ph, ze, ch = map(float, parts[3:6]); ga = float(parts[6])
                    sim.CZ_wave(q1, q2, ph, ze, ch, ga)
                else:
                    raise ValueError("cz_wave syntax: cz_wave q1 q2 phi [zeta chi gamma]")
            elif cmd == 'rxx':
                q1, q2, th = int(parts[1]), int(parts[2]), float(parts[3])
                sim.RXX(q1, q2, th)
            elif cmd == 'ryy':
                q1, q2, th = int(parts[1]), int(parts[2]), float(parts[3])
                sim.RYY(q1, q2, th)
            elif cmd == 'rzz':
                q1, q2, th = int(parts[1]), int(parts[2]), float(parts[3])
                sim.RZZ(q1, q2, th)
            elif cmd == 'cx':
                sim.CX(int(parts[1]), int(parts[2]))
            elif cmd == 'cy':
                sim.CY(int(parts[1]), int(parts[2]))
            elif cmd == 'cz':
                sim.CZ(int(parts[1]), int(parts[2]))
            elif cmd == 'ch':
                sim.CH(int(parts[1]), int(parts[2]))
            elif cmd == 'cs':
                sim.CS(int(parts[1]), int(parts[2]))
            elif cmd == 'ct':
                sim.CT(int(parts[1]), int(parts[2]))
            elif cmd == 'cp':
                sim.CP(int(parts[1]), int(parts[2]), float(parts[3]))
            elif cmd == 'crx':
                sim.CRX(int(parts[1]), int(parts[2]), float(parts[3]))
            elif cmd == 'cry':
                sim.CRY(int(parts[1]), int(parts[2]), float(parts[3]))
            elif cmd == 'crz':
                sim.CRZ(int(parts[1]), int(parts[2]), float(parts[3]))
            elif cmd in ('toffoli', 'ccx'):
                sim.Toffoli(int(parts[1]), int(parts[2]), int(parts[3]))
            elif cmd == 'cswap':
                sim.CSWAP(int(parts[1]), int(parts[2]), int(parts[3]))
            elif cmd == 'u':
                # u <q1,q2,...> <m*m complex entries>
                qs = _parse_q_list(parts[1])
                k = len(qs)
                m = 1 << k
                if len(parts) != 2 + m * m:
                    raise ValueError(f"u expects {m*m} matrix entries for k={k}")
                flat = [ _parse_complex_qasm(tok) for tok in parts[2:] ]
                U = np.array(flat, dtype=np.complex128).reshape(m, m)
                sim.apply_unitary(qs, U)
            elif cmd == 'u_full':
                N = 1 << sim.num_qubits
                if len(parts) != 1 + N * N:
                    raise ValueError(f"u_full expects {N*N} entries for N={N}")
                flat = [ _parse_complex_qasm(tok) for tok in parts[1:] ]
                U = np.array(flat, dtype=np.complex128).reshape(N, N)
                sim.apply_global_unitary_full(U)
            elif cmd == 'measure':
                q = int(parts[1])
                cbit = int(parts[2]) if len(parts) > 2 else None
                sim.measure(q, cbit)
            elif cmd == 'reset':
                sim.reset(int(parts[1]))
            # Noise channels
            elif cmd in ('noise_bitflip', 'nbf'):
                sim.noise_bit_flip(int(parts[1]), float(parts[2]))
            elif cmd in ('noise_phaseflip', 'npf'):
                sim.noise_phase_flip(int(parts[1]), float(parts[2]))
            elif cmd in ('noise_depolarizing', 'ndp'):
                sim.noise_depolarizing(int(parts[1]), float(parts[2]))
            elif cmd in ('noise_amp', 'nad'):
                sim.noise_amplitude_damping(int(parts[1]), float(parts[2]))
            elif cmd in ('noise_phase', 'nph'):
                sim.noise_phase_damping(int(parts[1]), float(parts[2]))
            # Utilities
            elif cmd == 'seed':
                sim.rng = np.random.default_rng(int(parts[1]))
            elif cmd == 'print_state':
                print(sim.state_ket())
            elif cmd == 'print_probs':
                p = sim.probs()
                for i, pr in enumerate(p):
                    if pr > 1e-12:
                        print(f"|{i:0{sim.num_qubits}b}>: {pr:.6f}")
            elif cmd == 'print_creg':
                for i, v in enumerate(sim.creg):
                    print(f"c[{i}] = {v}")
            elif cmd in ('barrier', 'delay'):
                pass
            else:
                print(f"Unknown command: {cmd}")
        except Exception as e:
            print(f"Error in '{line}': {e}")

    # Final state and creg
    print(sim.state_ket())
    for i, v in enumerate(sim.creg):
        print(f"c[{i}] = {v}")


