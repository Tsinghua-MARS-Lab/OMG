import numpy as np

from omg.benchmarks.metrics import diversity, matching_score, motion_fid, r_precision


def test_embedding_metrics_smoke():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16))
    b = rng.normal(size=(8, 16))
    assert motion_fid(a, b) >= 0
    assert diversity(a, num_pairs=4) > 0
    assert matching_score(a, b) > 0
    assert r_precision(a, b, top_k=3).shape == (3,)
