"""
Makes the repository root importable during tests.

Tests import project code as `from src.model.pipeline import ...`. With pytest's default
"prepend" import mode and no packaging config, pytest inserts the *test file's* directory
into sys.path -- not the repo root -- so `import src.*` fails under a bare `pytest`
invocation. `python -m pytest` happened to mask this, because -m prepends the cwd.

pytest inserts the directory containing the rootmost conftest.py into sys.path, so this
file existing here is what makes `src.*` resolve. It is intentionally (almost) empty --
do not delete it because it "looks unused".
"""
