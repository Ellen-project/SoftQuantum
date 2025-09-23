# SoftQuantum

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

**SoftQuantum** is a quantum computing simulation developed with Python and cuda.

he last code update is 2024/12/23.

## Core Features

* **Hybrid CPU/GPU Backend**: The simulator accelerates computations using CUDA kernels on compatible NVIDIA GPUs and seamlessly falls back to a multi-threaded NumPy implementation on systems without a dedicated GPU.
* **Interactive GUI (QASM Studio)**: A dedicated graphical interface provides an integrated development experience, allowing users to edit and execute QASM code and view results within a single application.
* **Syntax Highlighting**: The QASM Studio editor enhances code readability by automatically highlighting QASM keywords, numerical values, and comments.
* **Extensive Gate Library**: Supports a comprehensive set of standard single-qubit and multi-qubit gates, as well as advanced gates used in contemporary research, such as `fSim`, `RXX`, and `PhasedFSim`.
* **Extended QASM Syntax**: In addition to the standard QASM specification, the parser supports an extended instruction set with commands useful for debugging, such as `print_state` and `print_probs`.

## QASM Studio Interface



## Installation

### Prerequisites

* Python 3.7 or later
* NVIDIA GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-download) (required for GPU acceleration)
* A C++ compiler (e.g., GCC, Clang, or MSVC)
* Required Python libraries: `numpy`, `pybind11`

### Build Procedure

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/SoftQuantum.git](https://github.com/your-username/SoftQuantum.git)
    cd SoftQuantum
    ```

2.  **Install Python Dependencies**
    ```bash
    pip install numpy pybind11
    ```

3.  **Build the CUDA Extension Module (Optional)**
    To enable GPU acceleration, the `_svcuda` C++/CUDA extension module must be compiled. Execute the following command to build the module from the source code.

    ```bash
    python setup.py build_ext --inplace
    ```
    * Upon successful compilation, a `_svcuda` shared object file (`.so` on Linux, `.pyd` on Windows) will be created in the project directory.
    * If the CUDA Toolkit is not available or this step is skipped, the simulator will operate exclusively on the CPU backend.

## Usage

### 1. Executing the QASM Studio GUI

Launch the graphical user interface by running the following command from the terminal:

```bash
python qasm_gui.py
```


