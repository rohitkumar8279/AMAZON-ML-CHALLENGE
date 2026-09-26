#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "Amazon ML 2026 - V6 Environment Repair"
echo "============================================================"

# NON-NEGOTIABLE:
# - Do NOT modify/delete experiments/pre_submission_v4_real_frozen/
# - Do NOT modify V4 outputs
# - Do NOT run test inference
# - Do NOT create submission ZIP
# - Do NOT submit anything
# - Only repair/setup the V6 research environment

PROJECT_ROOT="$(pwd)"

echo
echo "[1/8] Current environment"
echo "PROJECT_ROOT=$PROJECT_ROOT"
command -v python || true
command -v python3 || true
command -v py || true

python --version 2>/dev/null || true
python3 --version 2>/dev/null || true
py --version 2>/dev/null || true

echo
echo "[2/8] Detect whether current Python is MSYS2"
CURRENT_PY="$(command -v python 2>/dev/null || true)"

if [[ -n "$CURRENT_PY" ]]; then
    echo "Current python: $CURRENT_PY"
fi

if [[ "$CURRENT_PY" == *"/msys"* ]] || \
   [[ "$CURRENT_PY" == *"/ucrt"* ]] || \
   [[ "$CURRENT_PY" == *"/mingw"* ]]; then
    echo "WARNING: current Python appears to be MSYS2/UCRT64."
else
    echo "Current Python does not appear to be MSYS2."
fi

echo
echo "[3/8] Look for official Windows CPython"
WIN_PY=""

if command -v py.exe >/dev/null 2>&1; then
    echo "Python launcher detected:"
    py.exe -0p || true

    # Prefer Python 3.13, then 3.12.
    if py.exe -3.13 -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
        WIN_PY="$(py.exe -3.13 -c 'import sys; print(sys.executable)')"
    elif py.exe -3.12 -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
        WIN_PY="$(py.exe -3.12 -c 'import sys; print(sys.executable)')"
    fi
fi

if [[ -n "$WIN_PY" ]]; then
    echo "Selected official CPython:"
    echo "$WIN_PY"
else
    echo "No suitable official Windows CPython 3.13/3.12 found."
fi

echo
echo "[4/8] Test official CPython native extensions"

WINDOWS_PYTHON_OK=0

if [[ -n "$WIN_PY" ]]; then
    if "$WIN_PY" -c "
import sys
print('Executable:', sys.executable)
import numpy
print('NumPy:', numpy.__version__)
import pandas
print('Pandas:', pandas.__version__)
" >/tmp/amazon_v6_python_test.txt 2>&1; then
        cat /tmp/amazon_v6_python_test.txt
        WINDOWS_PYTHON_OK=1
    else
        cat /tmp/amazon_v6_python_test.txt || true
        echo "Official CPython native-extension test failed."
    fi
fi

echo
echo "[5/8] Prefer WSL2 if available"

WSL_OK=0

if command -v wsl.exe >/dev/null 2>&1; then
    echo "WSL detected."
    wsl.exe --status || true
    echo
    wsl.exe -l -v || true

    if wsl.exe -l -q 2>/dev/null | grep -qi "Ubuntu"; then
        WSL_OK=1
        echo "Ubuntu WSL distribution detected."
    fi
else
    echo "WSL command not found."
fi

echo
echo "[6/8] Choose V6 environment"

# Strategy:
# 1. Prefer existing Ubuntu WSL for scientific/ML stack.
# 2. Otherwise use official Windows CPython if native extensions work.
# 3. Never use MSYS2 Python for V6.

if [[ "$WSL_OK" -eq 1 ]]; then

    echo "Using WSL2 Ubuntu for V6."

    V6_DIR_LINUX="$HOME/amazon_ml_v6"

    wsl.exe -d Ubuntu -- bash -lc "
        set -e

        mkdir -p ~/amazon_ml_v6

        if ! command -v python3 >/dev/null 2>&1; then
            echo 'python3 not installed in WSL'
            exit 20
        fi

        python3 --version

        if ! python3 -m venv --help >/dev/null 2>&1; then
            echo 'python3-venv package is missing.'
            echo 'Please install python3-venv in Ubuntu.'
            exit 21
        fi

        if [ ! -d ~/amazon_ml_v6/.venv ]; then
            python3 -m venv ~/amazon_ml_v6/.venv
        fi

        source ~/amazon_ml_v6/.venv/bin/activate

        python -m pip install --upgrade pip setuptools wheel

        python - <<'PY'
import sys
print('V6 Python:', sys.executable)
print('Python version:', sys.version)
PY

        python -m pip install \
            pandas \
            numpy \
            scipy \
            scikit-learn \
            rapidfuzz \
            lightgbm

        python - <<'PY'
print('Testing core V6 packages...')
import numpy
import pandas
import scipy
import sklearn
import rapidfuzz
import lightgbm

print('NumPy OK:', numpy.__version__)
print('Pandas OK:', pandas.__version__)
print('SciPy OK:', scipy.__version__)
print('Scikit-learn OK:', sklearn.__version__)
print('RapidFuzz OK')
print('LightGBM OK:', lightgbm.__version__)
print('ALL CORE PACKAGES OK')
PY
    "

    echo
    echo "WSL V6 environment created successfully."

elif [[ "$WINDOWS_PYTHON_OK" -eq 1 ]]; then

    echo "Using official Windows CPython."

    VENV_DIR="$PROJECT_ROOT/.venv-v6"

    if [[ ! -d "$VENV_DIR" ]]; then
        "$WIN_PY" -m venv "$VENV_DIR"
    fi

    # Convert Windows path to shell path when possible.
    if command -v cygpath >/dev/null 2>&1; then
        VENV_PY="$(cygpath -u "$VENV_DIR/Scripts/python.exe")"
    else
        VENV_PY="$VENV_DIR/Scripts/python.exe"
    fi

    "$VENV_PY" -m pip install --upgrade pip setuptools wheel

    "$VENV_PY" -m pip install \
        pandas \
        numpy \
        scipy \
        scikit-learn \
        rapidfuzz \
        lightgbm

    echo
    echo "Testing Windows V6 environment..."

    "$VENV_PY" -c "
import sys
print('Python:', sys.executable)
import numpy
import pandas
import scipy
import sklearn
import rapidfuzz
import lightgbm
print('NumPy:', numpy.__version__)
print('Pandas:', pandas.__version__)
print('SciPy:', scipy.__version__)
print('Scikit-learn:', sklearn.__version__)
print('RapidFuzz: OK')
print('LightGBM:', lightgbm.__version__)
"

    echo
    echo "CORE WINDOWS V6 ENVIRONMENT READY."

else

    echo
    echo "============================================================"
    echo "NO SAFE V6 ENVIRONMENT AVAILABLE"
    echo "============================================================"
    echo
    echo "Current MSYS2 Python must NOT be used for V6."
    echo
    echo "Recommended next step:"
    echo "Install/enable WSL2 Ubuntu or official CPython 3.13."
    echo
    echo "Do NOT modify the V4 environment."
    exit 30
fi

echo
echo "[7/8] Create V6 environment documentation"

mkdir -p experiments/v6_2

cat > experiments/v6_2/ENVIRONMENT.md <<'EOF'
# V6 Environment

## Rules

V4 REAL remains frozen and immutable.

V6 experiments must not modify:

experiments/pre_submission_v4_real_frozen/

No full test inference is performed during research experiments.

## Environment priority

1. WSL2 Ubuntu scientific environment
2. Official Windows CPython
3. Never MSYS2/UCRT64 Python for the ML/native-extension stack

## Required V6 components

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- RapidFuzz
- LightGBM

Future optional components:

- PyTorch
- FAISS
- transformers
- sentence-transformers

These should only be installed after the core environment passes.

EOF

echo
echo "[8/8] Final diagnostics"

echo
echo "Current shell:"
echo "$SHELL"

echo
echo "Windows Python:"
if [[ -n "${WIN_PY:-}" ]]; then
    echo "$WIN_PY"
fi

echo
echo "WSL:"
if [[ "$WSL_OK" -eq 1 ]]; then
    echo "Ubuntu WSL available"
else
    echo "Ubuntu WSL unavailable"
fi

echo
echo "============================================================"
echo "V6 ENVIRONMENT REPAIR COMPLETE"
echo "============================================================"
