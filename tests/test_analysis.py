
from src import analysis

def test_add():
    assert analysis.add(2, 3) == 5
    assert analysis.add(-1, 1) == 0