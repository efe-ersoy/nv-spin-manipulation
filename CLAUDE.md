# nv-spin-manipulation (package `nvsim`)

Pillar 2 of the magnetic-texture project: NV ground-state spin under pulse sequences, classical
magnetic noise, Monte Carlo ensembles and filter-function theory. See README.md for the API.

Project context, hypotheses and roadmap: `../nv-texture-imaging/docs/PLAN.md` (ask the user
to `/add-dir ../nv-texture-imaging` if you cannot read it). Data exchange with the other repos
follows `../nv-texture-imaging/docs/DATA_CONTRACT.md`.

## Rules

* `nvsim` stays generic: no sample-specific code, no imports from `magtex` or `nvti`.
  Sample-specific studies belong in `nv-texture-imaging`.
* Units inside `nvsim`: MHz, µs, mT (cyclic frequencies, no 2π). Convert at the boundary
  (files are SI).
* `nvsim.noise` sources implement `sample`, `autocorrelation`, `psd`, `lines`, `max_step`.
  The Gaussian filter-function theory is only valid for Gaussian sources; non-Gaussian sources
  must say so (`gaussian = False`) and theory code must refuse or warn for them.
* Tests: `python -m unittest discover tests` (keep the existing unittest style). Monte Carlo
  vs theory comparisons use statistical tolerances.
* Keep the README tables in sync when adding sources, sequences or public functions.
* Environment: this repo has its own `.venv` at the repo root. Install packages only there
  (never globally, never into a sibling repo's venv) and record every dependency in
  `pyproject.toml`. See PLAN §7 "Environments".
* Windows, VS Code; scripts run from the repo root (see .vscode).

## Task workflow

Tasks: `docs/tasks/T1xx-*.md` (template: `../nv-texture-imaging/docs/tasks/README.md`).
Read the brief, plan if non-trivial, implement, test, then append `## Report` to the brief
(what was done, results, deviations, open questions) and set `Status`. Commit with a message
starting with the task ID.
