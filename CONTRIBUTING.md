# Contributing to Project Residue

First off, thank you for considering contributing to Project Residue! It's people like you that make it an elite tool for the community.

## Code of Conduct

By participating in this project, you agree to abide by the terms of the MIT License and maintain a professional, respectful environment.

## How Can I Contribute?

### Reporting Bugs
- Use the [GitHub Issues](https://github.com/Orest-gt/residue/issues) to report bugs.
- Include details about your environment (CPU model, OS, Python version).
- Provide a minimal reproducible example.

### Suggesting Enhancements
- Open an issue with the [Enhancement] tag.
- Describe the performance bottleneck you are trying to solve.

### Pull Requests
1. Fork the repo and create your branch from `v4-baremetal`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (`pytest tests/`).
5. Make sure your code follows the existing style (we use `black` for Python and 2-space indentation for C++).

## Development Workflow

Project Residue uses a hybrid C++/Python stack. To set up your environment:

```bash
git clone https://github.com/Orest-gt/residue.git
cd residue
pip install -e .[dev]
```

Build the native extensions:
```bash
python setup.py build_ext --inplace
```

## Performance Standards

Every PR that touches the C++ core (`src/residue/core.cpp`) MUST be accompanied by a benchmark report using `tests/test_dispatch_benchmark.py`. We do not accept changes that introduce branch mispredictions or heap allocations in the hot path.

---
**Stay Bare-Metal. Stay Fast.** 🛡️
