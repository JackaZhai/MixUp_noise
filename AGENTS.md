# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the primary training entrypoint for Co-teaching runs.
- `model.py` defines the CNN architecture used in experiments.
- `loss.py` contains the Co-teaching loss implementation.
- `mixup_clean.py` provides an alternate training script for mixup/noise experiments.
- `data/` hosts dataset loaders (`mnist.py`, `cifar.py`) and shared utilities.
- `example.sh` shows a minimal run command; `results/` is created at runtime for logs.

## Build, Test, and Development Commands
- `python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5` runs a full training job.
- `python main.py --dataset mnist --noise_type pairflip --noise_rate 0.45` mirrors the example run in `example.sh`.
- No build system is defined; runs are direct Python invocations.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and module-level scripts rather than packages.
- Follow existing naming patterns: lowercase module names (`loss.py`), snake_case arguments (`noise_rate`), and all-caps dataset names in `data/` class names (e.g., `CIFAR10`).
- Keep changes compatible with the legacy environment noted in `README.md` (Python 2.7, PyTorch 0.3.0) unless explicitly modernizing.

## Testing Guidelines
- No automated test suite is present. Validate changes by running a short training job and confirming logs are written to `results/`.
- If you add tests, place them in a new `tests/` directory and document how to run them.

## Commit & Pull Request Guidelines
- Git history shows a single commit message (`first commit`), so there is no established convention.
- Use concise, imperative summaries (e.g., `Add CIFAR-100 noise option`) and include context in the body when changes are non-trivial.
- For pull requests, include a brief description, the exact command(s) run, and any sample metrics or logs produced.

## Configuration & Data
- Datasets download into `./data/` via the dataset classes; ensure the directory is writable.
- Key flags include `--dataset`, `--noise_type`, `--noise_rate`, and `--result_dir`.
