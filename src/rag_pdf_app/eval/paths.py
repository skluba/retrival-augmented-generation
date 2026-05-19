"""Repository-relative paths for bundled evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

# Bundled alongside pyproject / Docker WORKDIR copy (see Dockerfile).
EVAL_DATASET_CSV_NAME = "RAG_evaluation_dataset-convertcsv.csv"
_PYPROJECT = "pyproject.toml"


def repo_root() -> Path:
    """Tree root: nearest directory containing ``pyproject.toml``.

    Editable installs resolve to the git repo. Wheels in Docker resolve to ``/app`` because the
    image copies ``pyproject.toml`` there and the venv lives under ``/app/.venv`` (walking parents
    from ``site-packages`` reaches ``/app``). Using ``parents[3]`` from ``eval/paths.py`` breaks in
    installed layouts because that lands under ``.../python3.12``, not the project root.
    """

    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / _PYPROJECT).is_file():
            return d
    cwd = Path.cwd()
    for d in (cwd, *cwd.parents):
        if (d / _PYPROJECT).is_file():
            return d
    app = Path("/app")
    if (app / _PYPROJECT).is_file():
        return app
    raise RuntimeError(
        "Cannot locate project root (no pyproject.toml). Run from the repo or Docker WORKDIR, "
        "or pass --csv /path/to/" + EVAL_DATASET_CSV_NAME
    )


DEFAULT_IFC_EVAL_CSV = repo_root() / EVAL_DATASET_CSV_NAME
