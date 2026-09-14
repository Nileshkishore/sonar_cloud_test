import os
from testbook import testbook

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'analysis.ipynb'
)
print(f"NOTEBOOK_PATH: {NOTEBOOK_PATH}")
@testbook(NOTEBOOK_PATH, execute=True)
def test_add(tb):
    add = tb.ref("add")
    assert add(2, 3) == 5
    assert add(-1, 1) == 0