# Deltakit & QEC Numerical Benchmarks

Personal implementations, simulations, and benchmark tests based on Riverlane's Deltakit textbook:
https://textbook.riverlane.com/en/latest/index.html

## Contents
- **`src/repetition_code.py`**: Classical repetition code implementation covering:
  - Repetition encoding.
  - Independent bit-flip channel noise.
  - Majority-vote syndrome decoding.
  - Monte Carlo estimation of logical failure rates.
  - Correlated error burst model.
  - Benchmark plotting routines for threshold and error floors.
- **`tests/test_repetition_code.py`**: Automated unit tests verifying codeword parity, majority-vote logic, error suppression below threshold.

## Setup & Testing
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```
