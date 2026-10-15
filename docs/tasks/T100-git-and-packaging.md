# T100: Git and packaging for nvsim

Status: todo
Depends on: none
Repo: nv-spin-manipulation

## Goal
Put the repo under version control and make `nvsim` pip-installable, so the glue repo
(`nv-texture-imaging`) can `pip install -e ../nv-spin-manipulation`.

## Requirements
* `git init`; review `.gitignore` (keep `.venv/`, `output/`, `__pycache__/`; also ignore
  `.env` if it ever holds secrets; it currently only sets PYTHONPATH/PYTHONUTF8, so decide
  and note it). Initial commit of the current state, then a separate commit for this task.
* Minimal `pyproject.toml` (setuptools, flat layout with package `nvsim`, Python ≥ 3.11,
  dependencies as actually imported: numpy, scipy, matplotlib, qutip, …; check imports).
* Must not break the existing workflow: VS Code Run ▶ on `examples/` and `studies/` still works
  and `python -m unittest discover tests` still passes.
* README: short "Install" section.

## Acceptance
* In a fresh venv: `pip install -e .` then `python -c "import nvsim"` works.
* Existing tests pass.

## Report
