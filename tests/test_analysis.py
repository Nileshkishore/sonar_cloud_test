from testbook import testbook

@testbook('src/analysis.ipynb', execute=True)
def test_add(tb):
    add = tb.ref("add")
    assert add(2, 3) == 5
    assert add(-1, 1) == 0