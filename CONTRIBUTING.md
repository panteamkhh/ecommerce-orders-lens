# Contributing

Thanks for taking the time to contribute.

## Getting set up

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-dev.txt
pre-commit install          # optional but recommended
```

## Before you open a pull request

Run the same checks CI runs:

```bash
ruff check .
black --check .
pytest
python -m src.run_analysis   # make sure the pipeline still runs end to end
```

If `black` or `ruff --fix` reformat files, commit the result.

## Guidelines

- Keep business logic in `src/analysis.py` and presentation in
  `src/visualization.py` / `src/dashboard.py`; a chart should never compute its
  own numbers.
- Any new cleaning action must append an entry to the audit log via
  `log_step(...)` with a plain-language reason.
- Add or update a test in `tests/` for every behaviour change.
- Prefer small, focused commits with a clear message.

## Reporting issues

Open an issue describing the problem, the expected behaviour, and the command
or step that reproduces it.
