# T100: Git and packaging for nvsim

Status: done
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

**Done (2026-10-15).**

* `git init -b main`. Commit `ae5bed1` is the unchanged existing state (initial commit);
  the packaging changes and this report are a separate T100 commit.
* `.gitignore`: kept `.venv/`, `output/`, `__pycache__/`, `*.pyc`; added `*.egg-info/`,
  `build/`, `dist/` (artefacts of `pip install -e .`).
* `.env` decision: **tracked**. It only sets `PYTHONPATH=.` and `PYTHONUTF8=1` and is read by
  VS Code (`python.envFile`), so a fresh clone works out of the box. A comment in
  `.gitignore` says that secrets go into `.env.local`, which is ignored.
* `pyproject.toml`: setuptools (>= 68) build backend, flat layout with an explicit
  `packages = ["nvsim"]` (auto-discovery would refuse, because `examples/` and `studies/`
  have `__init__.py`), `requires-python = ">=3.11"`, version 0.1.0.
  Dependencies from the actual imports of `nvsim`: numpy, scipy, matplotlib, qutip, pillow
  (`PIL` is imported directly in `nvsim/bloch.py`). Lower bounds are the APIs the code relies
  on (numpy >= 2.0, scipy >= 1.11, matplotlib >= 3.8, qutip >= 5.0, pillow >= 10.0); the
  tested versions are noted in a comment.
* README: new "Install" section (venv, `pip install -e .`, install from a sibling repo).

**Checks**

* Fresh venv (base Python 3.14.8, outside the repo, deleted afterwards):
  `pip install -e .` resolved numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, qutip 5.3.1,
  pillow 12.3.0; `import nvsim` (plus `nvsim.study`, `nvsim.bloch`) works from a directory
  outside the repo with `PYTHONPATH` unset.
* `python -m unittest discover tests`: 52 tests OK in the fresh venv and in the repo `.venv`.
* VS Code workflow unchanged: with the repo `.venv` (nvsim *not* installed there) and
  `PYTHONPATH` = repo root, `examples/rabi_demo.py`, `studies/noise_study.py --quick` and
  `studies/compare_studies.py --only sample_linewidth` run without errors.

**Deviations / notes**

* The first fresh-venv attempt failed with Windows' 260-character path limit (numpy licence
  files inside a deep temp directory); the venv was recreated in a short path. Not a packaging
  problem, but worth knowing for venvs in deep folders (or enable long paths in Windows).
* Nothing was installed into the repo `.venv`; `nvsim` is not installed there because the
  scripts already work via `PYTHONPATH`. Running `pip install -e .` there is harmless if wanted.

**Open questions**

* PLAN §7 says each repo "pins" its dependencies. I used lower bounds (`>=`) so that
  `nv-texture-imaging` can share one resolver run with `magtex`; if exact pins are wanted,
  a lock file (e.g. `pip freeze > requirements-lock.txt`) would be the place for them.
* No remote is configured; pushing (and where to) is up to you.

