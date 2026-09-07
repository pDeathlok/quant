# CI runtime snapshots

`requirements/ci.txt` defines the source-regression dependency profile. It is not
the complete application, research, or deployment environment. The paired
`runtime-ci-py39.json` / `constraints-ci-py39.txt` and
`runtime-ci-py313.json` / `constraints-ci-py313.txt` record actual installed
versions, not estimated pins or a resolver's proposed installation.

## Observed environments

- `py39`: CPython 3.9.6, Darwin arm64, based on `/usr/bin/python3` and its
  existing system/user site packages. Missing CI tools were installed into
  `/private/tmp/quant-ci-remediation-py39` with `--system-site-packages`.
- `py313`: CPython 3.13.12, Darwin arm64, based on
  `/Users/didi/miniforge3/bin/python`. Missing CI tools were installed into
  `/private/tmp/quant-ci-remediation-py313` with `--system-site-packages`.

Neither base environment was modified. Temporary environments are local capture
provenance, not dependencies of the GitHub workflow. Snapshots contain only the
active dependency closure, interpreter/platform identity, and the SHA-256 of
`ci.txt`. They exclude unrelated installed packages, local paths, and index URLs.

## Capture and verify

From the repository root, choose an actual installed interpreter. In a disposable
venv, install `requirements/ci.txt`; add the matching `-c` constraints file to
reproduce an existing profile, or omit it for an intentional refresh. Then run:

```sh
"$PYTHON" scripts/ci/runtime_snapshot.py
"$PYTHON" scripts/ci/runtime_snapshot.py --format constraints
PYTHONPATH=src "$PYTHON" -m pytest tests/test_ci_runtime.py
```

Review and apply both outputs together to the matching JSON and constraints files.
Do not fill in a missing package version manually. Capture fails when an active
dependency is missing, violates a requirement, or uses a direct URL. Dependency
markers and requested extras are evaluated for the running interpreter; optional
extras not requested are excluded. Cycles terminate without skipping subsequent
version checks. Tests check constraints formatting, profile identity, root bounds,
and the requirements digest as well as traversal and error behavior.

## Validation limits

These are exact version constraints for observed macOS closures, not hashed wheel
locks or complete cross-platform locks. Linux-only dependencies (for example,
SQLAlchemy's platform-selected greenlet) can be resolved without pins. The Ubuntu
workflow installs the constraints, runs `pip check`, logs its actual closure, and
runs the selected source tests. A successful macOS capture does not establish
Linux wheel availability or a successful GitHub run. Python patch releases and
bootstrap pip are not pinned by the workflow. Full reproducibility would require
capturing and validating each target platform and locking its distribution hashes.

The workflow has read-only repository permissions, no persisted checkout
credentials, no deployment steps, a file market-data backend, empty SQL URL and
Tushare token, and temporary data roots. Python 3.9 excludes akquant via a marker;
the BYD regression using that dependency runs only on Python 3.13. This profile
does not certify the full production dependency set or the entire test suite.
