import math
import numpy as np
import pytest

try:
    from quantum_simulator_global import QuantumSimulator, execute_qasm, _HAVE_CUDA
except Exception as e:
    print("Import Error", f"quantum_simulator_global.py를 같은 폴더에 두세요.\n\n{e}")
    raise

I2 = np.eye(2, dtype=np.complex128)
X_GATE = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H_GATE = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
S_DAG_GATE = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
T_GATE = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
T_DAG_GATE = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)


def rx_matrix(theta):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry_matrix(theta):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta):
    a = theta / 2
    return np.array([[np.exp(-1j * a), 0], [0, np.exp(1j * a)]], dtype=np.complex128)


def u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=np.complex128)


def swap_matrix():
    return np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=np.complex128)


SWAP_GATE = swap_matrix()


def iswap_matrix():
    return np.array([[1, 0, 0, 0],
                     [0, 0, 1j, 0],
                     [0, 1j, 0, 0],
                     [0, 0, 0, 1]], dtype=np.complex128)


def iswap_theta_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[1, 0, 0, 0],
                     [0, c, 1j * s, 0],
                     [0, 1j * s, c, 0],
                     [0, 0, 0, 1]], dtype=np.complex128)


def fsim_matrix(theta, phi):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[1, 0, 0, 0],
                     [0, c, -1j * s, 0],
                     [0, -1j * s, c, 0],
                     [0, 0, 0, np.exp(-1j * phi)]], dtype=np.complex128)


def rxx_matrix(theta):
    a = theta / 2
    return math.cos(a) * np.eye(4, dtype=np.complex128) - 1j * math.sin(a) * np.kron(X_GATE, X_GATE)


def ryy_matrix(theta):
    a = theta / 2
    return math.cos(a) * np.eye(4, dtype=np.complex128) - 1j * math.sin(a) * np.kron(Y_GATE, Y_GATE)


def rzz_matrix(theta):
    a = theta / 2
    e = np.exp(-1j * a)
    ed = np.exp(1j * a)
    return np.diag([e, ed, ed, e]).astype(np.complex128)


def phased_fsim_matrix(theta, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.exp(-1j * (gamma + zeta)) * c, -1j * np.exp(-1j * (gamma - chi)) * s, 0.0],
        [0.0, -1j * np.exp(-1j * (gamma + chi)) * s, np.exp(-1j * (gamma - zeta)) * c, 0.0],
        [0.0, 0.0, 0.0, np.exp(-1j * (2 * gamma + phi))]
    ], dtype=np.complex128)


def phased_iswap_matrix(theta, phase):
    def rz(angle):
        return np.array([[np.exp(-1j * angle / 2), 0],
                         [0, np.exp(1j * angle / 2)]], dtype=np.complex128)

    pre = np.kron(rz(phase / 2), rz(-phase / 2))
    post = np.kron(rz(-phase / 2), rz(phase / 2))
    c = math.cos(theta)
    s = math.sin(theta)
    core = np.array([[1, 0, 0, 0],
                     [0, c, 1j * s, 0],
                     [0, 1j * s, c, 0],
                     [0, 0, 0, 1]], dtype=np.complex128)
    return pre @ core @ post


def random_state(num_qubits, seed):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=1 << num_qubits) + 1j * rng.normal(size=1 << num_qubits)
    vec = vec.astype(np.complex128)
    vec /= np.linalg.norm(vec)
    return vec


def random_unitary(dim, rng):
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(mat)
    diag = np.diag(r)
    phases = np.ones_like(diag)
    mask = np.abs(diag) > 1e-12
    phases[mask] = diag[mask] / np.abs(diag[mask])
    return (q * phases).astype(np.complex128)


def apply_unitary_reference(state, num_qubits, targets, U):
    t = tuple(targets)
    k = len(t)
    dim = 1 << num_qubits
    result = np.zeros(dim, dtype=np.complex128)
    for basis in range(dim):
        amp = state[basis]
        if abs(amp) < 1e-12:
            continue
        col = 0
        for idx, q in enumerate(t):
            col |= ((basis >> q) & 1) << idx
        for row in range(1 << k):
            new_basis = basis
            for idx, q in enumerate(t):
                bit = (row >> idx) & 1
                if bit:
                    new_basis |= (1 << q)
                else:
                    new_basis &= ~(1 << q)
            result[new_basis] += U[row, col] * amp
    return result


def apply_controlled_unitary_reference(state, num_qubits, controls, targets, U, ctrl_state=None):
    controls = tuple(controls)
    targets = tuple(targets)
    ctrl_state = tuple((1,) * len(controls) if ctrl_state is None else ctrl_state)
    dim = 1 << num_qubits
    result = np.zeros(dim, dtype=np.complex128)
    for basis in range(dim):
        amp = state[basis]
        if abs(amp) < 1e-12:
            continue
        ctrl_match = True
        for idx, c in enumerate(controls):
            bit = (basis >> c) & 1
            if bit != ctrl_state[idx]:
                ctrl_match = False
                break
        if ctrl_match:
            col = 0
            for idx, q in enumerate(targets):
                col |= ((basis >> q) & 1) << idx
            for row in range(1 << len(targets)):
                new_basis = basis
                for idx, q in enumerate(targets):
                    bit = (row >> idx) & 1
                    if bit:
                        new_basis |= (1 << q)
                    else:
                        new_basis &= ~(1 << q)
                result[new_basis] += U[row, col] * amp
        else:
            result[basis] += amp
    return result


def assert_state_almost_equal(actual, expected, tol=1e-9):
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    idx = None
    for i, val in enumerate(expected):
        if abs(val) > tol:
            idx = i
            break
    if idx is not None and abs(expected[idx]) > tol and abs(actual[idx]) > tol:
        phase = actual[idx] / expected[idx]
        actual = actual / phase
    assert np.allclose(actual, expected, atol=tol)


def basis_state(num_qubits, index):
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[index] = 1.0
    return state


SINGLE_QUBIT_CASES = [
    ("I", lambda params: I2, {}),
    ("H", lambda params: H_GATE, {}),
    ("S", lambda params: S_GATE, {}),
    ("Sdg", lambda params: S_DAG_GATE, {}),
    ("T", lambda params: T_GATE, {}),
    ("Tdg", lambda params: T_DAG_GATE, {}),
    ("X", lambda params: X_GATE, {}),
    ("Y", lambda params: Y_GATE, {}),
    ("Z", lambda params: Z_GATE, {}),
    ("RX", lambda params: rx_matrix(params["theta"]), {"theta": 0.37}),
    ("RY", lambda params: ry_matrix(params["theta"]), {"theta": -0.58}),
    ("RZ", lambda params: rz_matrix(params["theta"]), {"theta": 1.12}),
    ("U", lambda params: u3_matrix(params["theta"], params["phi"], params["lam"]),
     {"theta": 0.42, "phi": -0.31, "lam": 0.27}),
]


@pytest.mark.parametrize("method,matrix_fn,gate_kwargs", SINGLE_QUBIT_CASES)
@pytest.mark.parametrize("target", [0, 1, 2])
def test_single_qubit_gates_match_reference(method, matrix_fn, gate_kwargs, target):
    num_qubits = 3
    sim = QuantumSimulator(num_qubits, seed=123)
    state = random_state(num_qubits, seed=500 + target * 11 + len(method))
    sim.state = state.copy()
    kwargs = dict(gate_kwargs)
    getattr(sim, method)(target, **kwargs)
    expected = apply_unitary_reference(state, num_qubits, [target], matrix_fn(kwargs))
    assert_state_almost_equal(sim.state, expected)


TWO_QUBIT_CASES = [
    ("SWAP", lambda params: SWAP_GATE, {}),
    ("ISWAP", lambda params: iswap_matrix(), {}),
    ("ISWAP_theta", lambda params: iswap_theta_matrix(params["theta"]), {"theta": 0.63}),
    ("ISWAP_pow", lambda params: iswap_theta_matrix(params["p"] * math.pi / 2), {"p": 0.4}),
    ("ISWAPdg", lambda params: iswap_theta_matrix(-math.pi / 2), {}),
    ("fSim", lambda params: fsim_matrix(params["theta"], params["phi"]), {"theta": 0.27, "phi": -0.19}),
    ("SYC", lambda params: fsim_matrix(math.pi / 2, math.pi / 6), {}),
    ("RXX", lambda params: rxx_matrix(params["theta"]), {"theta": 0.77}),
    ("RYY", lambda params: ryy_matrix(params["theta"]), {"theta": -0.58}),
    ("RZZ", lambda params: rzz_matrix(params["theta"]), {"theta": 1.05}),
    ("PhasedFSim", lambda params: phased_fsim_matrix(params["theta"], params["zeta"], params["chi"], params["gamma"], params["phi"]),
        {"theta": 0.33, "zeta": 0.12, "chi": -0.21, "gamma": 0.4, "phi": -0.17}),
    ("CZ_wave", lambda params: phased_fsim_matrix(0.0, params["zeta"], params["chi"], params["gamma"], params["phi"]),
        {"phi": 0.28, "zeta": -0.11, "chi": 0.09, "gamma": 0.07}),
    ("PhasedISWAP", lambda params: phased_iswap_matrix(params["theta"], params["phase"]),
        {"theta": 0.41, "phase": 0.30}),
]


@pytest.mark.parametrize("method,matrix_fn,gate_kwargs", TWO_QUBIT_CASES)
@pytest.mark.parametrize("targets", [(0, 1), (0, 2), (1, 2)])
def test_two_qubit_gates_match_reference(method, matrix_fn, gate_kwargs, targets):
    num_qubits = 3
    sim = QuantumSimulator(num_qubits, seed=321)
    state = random_state(num_qubits, seed=700 + targets[0] * 5 + targets[1])
    sim.state = state.copy()
    kwargs = dict(gate_kwargs)
    getattr(sim, method)(*targets, **kwargs)
    expected = apply_unitary_reference(state, num_qubits, list(targets), matrix_fn(kwargs))
    assert_state_almost_equal(sim.state, expected)


ONE_CONTROL_GATES = [
    ("CX", lambda params: X_GATE, {}),
    ("CY", lambda params: Y_GATE, {}),
    ("CZ", lambda params: Z_GATE, {}),
    ("CH", lambda params: H_GATE, {}),
    ("CS", lambda params: S_GATE, {}),
    ("CT", lambda params: T_GATE, {}),
    ("CP", lambda params: np.array([[1, 0], [0, np.exp(1j * params["lam"])]] , dtype=np.complex128), {"lam": 0.23}),
    ("CRX", lambda params: rx_matrix(params["theta"]), {"theta": -0.67}),
    ("CRY", lambda params: ry_matrix(params["theta"]), {"theta": 0.45}),
    ("CRZ", lambda params: rz_matrix(params["theta"]), {"theta": 1.11}),
]


@pytest.mark.parametrize("method,matrix_fn,gate_kwargs", ONE_CONTROL_GATES)
@pytest.mark.parametrize("control,target", [(0, 1), (2, 0)])
def test_single_controlled_gates_match_reference(method, matrix_fn, gate_kwargs, control, target):
    num_qubits = 3
    sim = QuantumSimulator(num_qubits, seed=222)
    state = random_state(num_qubits, seed=800 + control * 13 + target)
    sim.state = state.copy()
    kwargs = dict(gate_kwargs)
    getattr(sim, method)(control, target, **kwargs)
    expected = apply_controlled_unitary_reference(state, num_qubits, [control], [target], matrix_fn(kwargs))
    assert_state_almost_equal(sim.state, expected)


def test_cu_gate_match_reference():
    num_qubits = 3
    sim = QuantumSimulator(num_qubits, seed=333)
    state = random_state(num_qubits, seed=915)
    sim.state = state.copy()
    rng = np.random.default_rng(77)
    U = random_unitary(2, rng)
    control, target = 2, 1
    sim.CU(control, target, U)
    expected = apply_controlled_unitary_reference(state, num_qubits, [control], [target], U)
    assert_state_almost_equal(sim.state, expected)


@pytest.mark.parametrize("controls,target", [((0, 1), 2), ((2, 1), 0)])
def test_toffoli_gate_match_reference(controls, target):
    num_qubits = 3
    sim = QuantumSimulator(num_qubits, seed=444)
    state = random_state(num_qubits, seed=905 + target)
    sim.state = state.copy()
    sim.Toffoli(controls[0], controls[1], target)
    expected = apply_controlled_unitary_reference(state, num_qubits, list(controls), [target], X_GATE)
    assert_state_almost_equal(sim.state, expected)


@pytest.mark.parametrize("control,targets", [(0, (1, 2)), (2, (0, 1))])
def test_cswap_gate_match_reference(control, targets):
    num_qubits = 3
    sim = QuantumSimulator(num_qubits, seed=555)
    state = random_state(num_qubits, seed=940 + control)
    sim.state = state.copy()
    sim.CSWAP(control, targets[0], targets[1])
    expected = apply_controlled_unitary_reference(state, num_qubits, [control], list(targets), SWAP_GATE)
    assert_state_almost_equal(sim.state, expected)


def test_apply_controlled_unitary_with_zero_control_state():
    num_qubits = 3
    sim = QuantumSimulator(num_qubits, seed=666)
    state = random_state(num_qubits, seed=999)
    sim.state = state.copy()
    controls = [2]
    targets = [0]
    ctrl_state = (0,)
    U = H_GATE
    sim.apply_controlled_unitary(controls, targets, U, ctrl_state=ctrl_state)
    expected = apply_controlled_unitary_reference(state, num_qubits, controls, targets, U, ctrl_state=ctrl_state)
    assert_state_almost_equal(sim.state, expected)


def test_apply_unitary_three_qubit_gate():
    num_qubits = 3
    sim = QuantumSimulator(num_qubits, seed=777)
    state = random_state(num_qubits, seed=1001)
    sim.state = state.copy()
    rng = np.random.default_rng(101)
    targets = [0, 1, 2]
    U = random_unitary(1 << len(targets), rng)
    sim.apply_unitary(targets, U)
    expected = apply_unitary_reference(state, num_qubits, targets, U)
    assert_state_almost_equal(sim.state, expected)


def test_apply_unitary_respects_target_order():
    num_qubits = 2
    sim = QuantumSimulator(num_qubits, seed=888)
    state = random_state(num_qubits, seed=1102)
    sim.state = state.copy()
    rng = np.random.default_rng(202)
    targets = [1, 0]
    U = random_unitary(1 << len(targets), rng)
    sim.apply_unitary(targets, U)
    expected = apply_unitary_reference(state, num_qubits, targets, U)
    assert_state_almost_equal(sim.state, expected)


def test_measure_updates_state_and_creg():
    sim = QuantumSimulator(2, seed=900)
    sim.creg = [0, 0]
    state = basis_state(2, 3)
    sim.state = state.copy()
    outcome = sim.measure(0, cbit=1)
    assert outcome == 1
    assert sim.creg[1] == 1
    assert_state_almost_equal(sim.state, state)


def test_measure_all_returns_bits_in_lsb_order():
    sim = QuantumSimulator(3, seed=901)
    state = basis_state(3, 5)
    sim.state = state.copy()
    bits = sim.measure_all()
    assert bits == [1, 0, 1]
    assert_state_almost_equal(sim.state, state)


def test_reset_projects_target_to_zero():
    sim = QuantumSimulator(2, seed=902)
    state = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    state /= np.linalg.norm(state)
    sim.state = state.copy()
    sim.reset(0)
    expected = basis_state(2, 0)
    assert_state_almost_equal(sim.state, expected)


def test_apply_global_unitary_full_matches_reference():
    num_qubits = 2
    sim = QuantumSimulator(num_qubits, seed=903)
    state = random_state(num_qubits, seed=1203)
    sim.state = state.copy()
    rng = np.random.default_rng(303)
    U = random_unitary(1 << num_qubits, rng)
    sim.apply_global_unitary_full(U)
    expected = U @ state
    assert_state_almost_equal(sim.state, expected)


def test_noise_bit_flip_probability_one():
    sim = QuantumSimulator(1, seed=904)
    sim.state = basis_state(1, 0)
    sim.noise_bit_flip(0, 1.0)
    expected = basis_state(1, 1)
    assert_state_almost_equal(sim.state, expected)


def test_noise_phase_flip_on_plus_state():
    sim = QuantumSimulator(1, seed=905)
    plus = np.array([1.0, 1.0], dtype=np.complex128)
    plus /= np.linalg.norm(plus)
    sim.state = plus.copy()
    sim.noise_phase_flip(0, 1.0)
    expected = np.array([1.0, -1.0], dtype=np.complex128)
    expected /= np.linalg.norm(expected)
    assert_state_almost_equal(sim.state, expected)


def test_noise_depolarizing_identity_for_zero_probability():
    sim = QuantumSimulator(1, seed=906)
    state = random_state(1, seed=140)
    sim.state = state.copy()
    sim.noise_depolarizing(0, 0.0)
    assert_state_almost_equal(sim.state, state)


def test_noise_depolarizing_probability_one_gives_pauli_state():
    sim = QuantumSimulator(1, seed=907)
    sim.state = basis_state(1, 0)
    sim.noise_depolarizing(0, 1.0)
    probs = np.abs(sim.state) ** 2
    assert any(np.allclose(probs, target, atol=1e-8) for target in (np.array([1.0, 0.0]), np.array([0.0, 1.0])))
    assert abs(np.linalg.norm(sim.state) - 1.0) < 1e-12


def test_noise_amplitude_damping_probability_one():
    sim = QuantumSimulator(1, seed=908)
    sim.state = basis_state(1, 1)
    sim.noise_amplitude_damping(0, 1.0)
    expected = basis_state(1, 0)
    assert_state_almost_equal(sim.state, expected)


def test_noise_phase_damping_extremes():
    sim = QuantumSimulator(1, seed=909)
    plus = np.array([1.0, 1.0], dtype=np.complex128)
    plus /= np.linalg.norm(plus)
    sim.state = plus.copy()
    sim.noise_phase_damping(0, 0.0)
    assert_state_almost_equal(sim.state, plus)
    sim.state = plus.copy()
    sim.noise_phase_damping(0, 1.0)
    probs = np.abs(sim.state) ** 2
    assert any(np.allclose(probs, target, atol=1e-8) for target in (np.array([1.0, 0.0]), np.array([0.0, 1.0])))




