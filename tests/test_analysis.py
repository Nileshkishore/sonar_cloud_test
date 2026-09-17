
"""
Notebook-based test for the `analysis.ipynb` notebook.

This test uses the `testbook` library to execute a Jupyter notebook and
access objects defined inside it.

Key terms:
- `testbook`: a testing library that runs notebooks in an isolated IPython
    kernel and exposes a `@testbook` decorator for tests.
- `@testbook(path, execute=True)`: decorator that runs the notebook at
    `path` and provides a `tb` (TestbookNotebook) object to the test function.
- `tb` (the test function argument): the Testbook notebook object. Use
    `tb.ref('name')` to obtain a reference to a top-level object defined in the
    notebook (for example, a function named `add`).

Notes:
- `testbook` executes notebook cells in a separate kernel process. Coverage
    tools running in the main pytest process will not automatically record
    notebook-executed code unless subprocess coverage support is enabled
    (COVERAGE_PROCESS_START + sitecustomize). In CI, ensure `jupyter` and
    `testbook` are installed and coverage subprocess support is configured if
    you want notebook execution included in `coverage.xml`.
"""

import os
from testbook import testbook

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis.ipynb')


@testbook(NOTEBOOK_PATH, execute=True)
def test_add(tb):
    add = tb.ref('add')
    assert add(2, 3) == 5
    assert add(-1, 1) == 0