
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

from unittest.mock import Mock

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis.ipynb')


@testbook(NOTEBOOK_PATH, execute=True)
def test_add(tb):
    add = tb.ref('add')
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

@testbook(NOTEBOOK_PATH, execute=True)
def test_add_with_mock(tb):
    # get the real `add` function from the executed notebook
    real_add = tb.ref('add')
    
    # create a mock that wraps the real function so calls are recorded
    # while still executing the real implementation
    mock_add = Mock(wraps=real_add)

    # optionally make tb.ref return the mocked wrapper (keeps same call site)
    tb.ref = Mock(return_value=mock_add)

    add = tb.ref('add')
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

    # ensure the mock wrapper was called and forwarded to the real function
    mock_add.assert_called()
    mock_add.assert_any_call(2, 3)
    mock_add.assert_any_call(-1, 1)