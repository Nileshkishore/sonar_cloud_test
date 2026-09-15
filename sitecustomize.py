import os

# Enable coverage for subprocesses (kernels) when COVERAGE_PROCESS_START is set.
# This file will be found if the repository root is added to PYTHONPATH in CI.
if os.environ.get('COVERAGE_PROCESS_START'):
    try:
        import coverage
        coverage.process_startup()
    except Exception:
        # best-effort; don't fail if coverage isn't available
        pass
