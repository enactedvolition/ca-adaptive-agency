from mvp import smoke

def test_smoke(tmp_path):
    assert smoke.main(["--steps", "40", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "frame.png").stat().st_size > 1000
